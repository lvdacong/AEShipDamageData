"""
AE_model_train_and_detect.py
=============================
Scenario-based AE training and damage detection orchestrator.
Reads AE_settings.json, runs pretrain/TL/ablation per scenario.

Usage:
    cd script && python AE_model_train_and_detect.py
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# 导入基础辅助模块
import sys
sys.path.insert(0, os.path.dirname(__file__))
from AE_train_model_auxiliary import (
    Autoencoder, load_data_from_path, train_model,
    FIG_DPI, PLOT_STYLE, apply_style
)
from AE_model_train_and_detect_auxiliary import HAS_PYVISTA

# 导入实验辅助模块
from AE_freeze_ablation_auxiliary import run_freeze_ablation
from AE_tl_comparison_auxiliary import run_tl_comparison


def load_settings(settings_path: Optional[str] = None) -> dict:
    """
    加载配置文件

    Args:
        settings_path: 配置文件路径，默认为脚本同目录下的AE_settings.json

    Returns:
        配置字典
    """
    if settings_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        settings_path = os.path.join(script_dir, 'AE_settings.json')

    if not os.path.exists(settings_path):
        raise FileNotFoundError(f"配置文件不存在: {settings_path}")

    with open(settings_path, 'r', encoding='utf-8') as f:
        settings = json.load(f)

    print(f"[配置] 加载配置文件: {settings_path}")
    return settings


def resolve_paths(settings: dict) -> dict:
    """
    解析配置中的相对路径为绝对路径

    Args:
        settings: 原始配置字典

    Returns:
        路径解析后的配置字典
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

    # 解析workspace_dir
    if settings['global']['workspace_dir'] == 'auto':
        settings['global']['workspace_dir'] = workspace_dir

    return settings


def get_or_train_pretrain_model(
    scenario_name: str,
    scenario_config: dict,
    settings: dict,
    device: torch.device
) -> Tuple[str, str]:
    """
    获取或训练预训练模型

    如果预训练模型已存在，直接返回路径；否则训练新模型。
    支持跨场景复用：如果多个场景使用相同的old_data_folder，
    只训练一次预训练模型。

    Args:
        scenario_name: 场景名称
        scenario_config: 场景配置
        settings: 全局设置
        device: 计算设备

    Returns:
        (pretrain_pth, pretrain_dir): 预训练模型路径和目录
    """
    workspace_dir = settings['global']['workspace_dir']
    output_base = os.path.join(workspace_dir, settings['global']['output_dir'])
    preprocess_base = os.path.join(workspace_dir, settings['global']['preprocess_output_base'])

    old_data_folder = scenario_config['old_data_folder']

    # 预训练模型目录
    pretrain_dir = os.path.join(output_base, scenario_name, 'pretrain')
    pretrain_pth = os.path.join(pretrain_dir, 'autoencoder.pth')
    pretrain_losses = os.path.join(pretrain_dir, 'training_losses.csv')

    # 检查是否已存在（pretrain 跨场景共享，避免重复训练）
    if os.path.exists(pretrain_pth) and os.path.exists(pretrain_losses):
        print(f"\n[预训练] 模型已存在: {pretrain_pth}")
        return pretrain_pth, pretrain_dir

    # 搜索其他场景是否有相同old_data的预训练模型
    for other_scenario in settings['scenarios']:
        if other_scenario == scenario_name:
            continue
        other_config = settings['scenarios'][other_scenario]
        if other_config['old_data_folder'] == old_data_folder:
            other_pretrain_dir = os.path.join(output_base, other_scenario, 'pretrain')
            other_pretrain_pth = os.path.join(other_pretrain_dir, 'autoencoder.pth')
            if os.path.exists(other_pretrain_pth):
                print(f"\n[预训练] 复用场景 '{other_scenario}' 的预训练模型")
                os.makedirs(pretrain_dir, exist_ok=True)
                return other_pretrain_pth, other_pretrain_dir

    # 需要训练新的预训练模型
    print(f"\n{'='*60}")
    print(f"[{scenario_name}] 训练预训练模型")
    print(f"{'='*60}")

    os.makedirs(pretrain_dir, exist_ok=True)

    # 加载预训练数据
    old_data_path = os.path.join(preprocess_base, old_data_folder)
    V_train, V_val = load_data_from_path(
        old_data_path,
        settings['training']['val_samples'],
        "预训练数据"
    )

    # 训练模型
    ae_config = settings['model']
    training_config = settings['training']

    model, train_losses, val_losses = train_model(
        V_train, V_val, device,
        epochs=training_config['pretrain']['epochs'],
        lr=training_config['pretrain']['lr'],
        batch_size=training_config['pretrain']['batch_size'],
        output_dir=pretrain_dir,
        ae_config=ae_config,
        training_config=training_config,
        plot_style=PLOT_STYLE,
        fig_dpi=FIG_DPI,
        pretrain_model_path=None,
        model_name="Pre-train",
        train_shuffle=training_config.get('shuffle', True)
    )

    print(f"[预训练] 模型保存: {pretrain_pth}")
    return pretrain_pth, pretrain_dir


def process_scenario(
    scenario_name: str,
    scenario_config: dict,
    settings: dict,
    device: torch.device
) -> None:
    """
    处理单个迁移场景的所有实验

    Args:
        scenario_name: 场景名称
        scenario_config: 场景配置
        settings: 全局设置
        device: 计算设备
    """
    print(f"\n{'#'*70}")
    print(f"# 场景: {scenario_config.get('name', scenario_name)}")
    print(f"# 描述: {scenario_config.get('description', 'N/A')}")
    print(f"{'#'*70}")

    # 1. 确保预训练模型存在
    pretrain_pth, pretrain_dir = get_or_train_pretrain_model(
        scenario_name, scenario_config, settings, device
    )

    # 2. 冻结策略消融实验（如果启用）
    freeze_config = scenario_config.get('freeze_ablation', {})
    if freeze_config.get('enabled', False):
        try:
            run_freeze_ablation(
                scenario_name=scenario_name,
                scenario_config=scenario_config,
                pretrain_pth=pretrain_pth,
                settings=settings,
                device=device
            )
        except Exception as e:
            print(f"[错误] 冻结策略消融实验失败: {e}")
            import traceback
            traceback.print_exc()

    # 3. TL效果对比实验（如果启用）
    tl_config = scenario_config.get('tl_comparison', {})
    if tl_config.get('enabled', False):
        try:
            run_tl_comparison(
                scenario_name=scenario_name,
                scenario_config=scenario_config,
                pretrain_pth=pretrain_pth,
                settings=settings,
                device=device
            )
        except Exception as e:
            print(f"[错误] TL效果对比实验失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[完成] 场景 '{scenario_name}' 所有实验完成")


def main(settings_path: Optional[str] = None):
    """
    主入口函数

    Args:
        settings_path: 配置文件路径（可选）
    """
    print("\n" + "="*70)
    print("AE 模型训练 + 损伤检测")
    print("="*70)

    # 加载配置
    settings = load_settings(settings_path)
    settings = resolve_paths(settings)

    # 设置随机种子
    seed = settings['global']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 设置设备
    device_str = settings['global']['device']
    if device_str == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"[设备] {device}")
    print(f"[随机种子] {seed}")
    print(f"[PyVista] {'可用' if HAS_PYVISTA else '不可用（跳过3D渲染）'}")

    # 创建输出目录
    workspace_dir = settings['global']['workspace_dir']
    output_dir = os.path.join(workspace_dir, settings['global']['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    print(f"[输出目录] {output_dir}")

    # 显示场景概览
    scenarios = settings['scenarios']
    print(f"\n[场景] 共 {len(scenarios)} 个迁移场景:")
    for name, config in scenarios.items():
        print(f"  - {name}: {config.get('name', 'N/A')}")

    # 处理每个场景
    for scenario_name, scenario_config in scenarios.items():
        process_scenario(scenario_name, scenario_config, settings, device)

    print("\n" + "="*70)
    print("所有场景处理完成！")
    print("="*70)


if __name__ == "__main__":
    main()
