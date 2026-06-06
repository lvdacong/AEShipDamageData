"""
AD_preprocess_datasets.py
==========================
Preprocess all SHM datasets: extract measurement-point stress from
raw NPY simulation outputs and save as preprocessed_data_raw.npz.

Usage:
    cd script && python AD_preprocess_datasets.py
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import stat
import sys
from dataclasses import dataclass
from typing import Callable, Literal, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

# ========================================
# ========================================
# 数据集配置
# ========================================
# 【配置来源说明】
#   - 通过 AAA_oneclick_run.py 调用时：从 TL_settings.jsonc 读取
#     (FEM_models + simulation_counts + cases + offset)
#   - 独立运行本脚本时：使用下方默认配置
#
# ========================================

# 数据集配置（独立运行时的默认配置）
# 通过 run_from_config 调用时会被 TL_settings.jsonc 配置覆盖
DATASET_CONFIGS = {
    # 此配置仅供独立运行使用，实际运行时应通过 TL_settings.jsonc 配置
}

# ========================================
# 全局配置
# ========================================

# --- 1. 路径配置 ---
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, 'script', 'AD_preprocess_datasets_output')

# --- 2. 数据处理参数 ---
MEASURES_ID_CSV = '../script/AC_convert_and_extract_output/measures_ID_original.csv'
RECONSTRUCT_COLUMN = 'all_measures'
USE_NPY = True
SEED = 42

# --- 3. 处理流程开关 ---
DO_COLLECT = True
DO_VISUALIZE = True

# --- 5. 已知FEM模型名称（可被run_from_config动态扩展） ---
KNOWN_MODELS = ['health', 'first_damage', 'second_damage', 'damage_repaired']

# --- 4. Case配置列表 ---
# 【配置来源说明】
#   - 通过 AAA_oneclick_run.py 调用时：从 TL_settings.jsonc 的 cases 区动态生成
#   - 输出文件夹命名规则：{fem_structure}_{measures_suffix}_{sample_count}
#     例如：first_damage_offset_100 表示使用first_damage结构、offset测点、100个样本
#
# ========================================

# Case配置（独立运行时的默认配置）
# 通过 run_from_config 调用时会被 TL_settings.jsonc 配置覆盖
CASE_CONFIGS = [
    # 此配置仅供独立运行使用，实际运行时应通过 TL_settings.jsonc 配置
]


# --- 5. 可视化参数 ---
PREPROCESS_PLOT_FIRST_N = 25
SHOW_PLOT_PROGRESS = True
PREPROCESS_GRID_ROWS = 5
PREPROCESS_GRID_COLS = 5

# 可视化样式参数
PREPROCESS_SCATTER_COLOR = 'tab:blue'
PREPROCESS_DENSITY_COLOR = 'tab:orange'
PREPROCESS_SCATTER_SIZE = 1.5
PREPROCESS_SHOW_GRID = False
PREPROCESS_KDE_LINEWIDTH = 0.5
PREPROCESS_HIST_BINS = 100
PREPROCESS_HIST_ALPHA = 0.20

# 图像输出参数
FIG_DPI = 300

# 绘图风格（学术风）
PLOT_STYLE = {
    "font.family": "Times New Roman",
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.grid": False,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.bottom": True,
    "axes.spines.left": True,
}

# ========================================
# 内嵌工具函数：通用
# ========================================

def is_nested_offset(offset_value) -> bool:
    """
    检测offset配置是否为嵌套数组（批量offset方案）

    单个方案: [0] 或 [1,2,3] - 元素全是int
    批量方案: [[0],[1],[2]] 或 [[1,2],[3,4]] - 元素全是list

    Args:
        offset_value: offset配置值

    Returns:
        True 表示嵌套数组（批量方案），False 表示单个方案或无效
    """
    if not isinstance(offset_value, list) or len(offset_value) == 0:
        return False
    return isinstance(offset_value[0], list)


def expand_case_offsets(case: dict) -> list:
    """
    展开case的offset配置

    如果offset是嵌套数组（如[[0],[1],[2]]），则展开为多个独立的case。
    如果offset是单个数组（如[0]）或无offset，则返回原始case。

    Args:
        case: case配置字典

    Returns:
        展开后的case列表
    """
    offset_value = case.get('offset')

    if not offset_value:
        return [case]

    if is_nested_offset(offset_value):
        expanded_cases = []
        for single_offset in offset_value:
            expanded_case = case.copy()
            expanded_case['offset'] = single_offset
            expanded_cases.append(expanded_case)
        return expanded_cases
    else:
        return [case]


def expand_all_cases(config: dict) -> list:
    """
    展开所有cases的offset和drift配置

    Args:
        config: 全局配置字典

    Returns:
        展开后的所有cases列表
    """
    cases = config.get('cases', [])
    all_expanded = []
    for case in cases:
        # 先展开offset
        offset_expanded = expand_case_offsets(case)
        # 再展开drift
        for expanded_case in offset_expanded:
            all_expanded.extend(expand_case_drifts(expanded_case))
    return all_expanded


def get_offset_name(offset_indices: list) -> str:
    """
    根据偏移测点索引列表自动生成偏移方案名称

    命名规则:
    - [2] -> "offset_2"
    - [2, 3, 4] -> "offset_2to4"
    - [1, 3, 5] -> "offset_1_3_5" (非连续)
    - None 或 [] -> "none"

    Args:
        offset_indices: 偏移测点行索引列表，如 [2] 或 [2, 3, 4]

    Returns:
        偏移方案名称字符串
    """
    if offset_indices is None or len(offset_indices) == 0:
        return "none"

    sorted_indices = sorted(offset_indices)

    if len(sorted_indices) == 1:
        return f"offset_{sorted_indices[0]}"

    # 检查是否连续
    is_continuous = all(
        sorted_indices[i+1] - sorted_indices[i] == 1
        for i in range(len(sorted_indices) - 1)
    )

    if is_continuous:
        return f"offset_{sorted_indices[0]}to{sorted_indices[-1]}"
    else:
        # 非连续情况，用下划线连接
        return "offset_" + "_".join(str(i) for i in sorted_indices)


def get_offset_measures_filename(offset_indices: list) -> str:
    """
    生成偏移测点CSV文件名

    Args:
        offset_indices: 偏移测点行索引列表

    Returns:
        文件名，如 "measures_ID_offset_2.csv" 或 "measures_ID_offset_2to4.csv"
    """
    offset_name = get_offset_name(offset_indices)
    if offset_name == "none":
        return "measures_ID_offset.csv"  # fallback
    return f"measures_ID_{offset_name}.csv"


# ========================================
# 内嵌工具函数：温漂(Drift)相关
# ========================================

def is_nested_drift(drift_value) -> bool:
    """
    检测drift配置是否为嵌套数组（批量drift方案）

    单个方案: [0] 或 [1,2,3] - 元素全是int
    批量方案: [[0],[1],[2]] 或 [[1,2],[3,4]] - 元素全是list
    字符串 "all": 全通道漂移，非嵌套

    Args:
        drift_value: drift配置值

    Returns:
        True 表示嵌套数组（批量方案），False 表示单个方案或无效
    """
    if isinstance(drift_value, str):
        return False
    if not isinstance(drift_value, list) or len(drift_value) == 0:
        return False
    return isinstance(drift_value[0], list)


def get_drift_name(drift_indices) -> str:
    """
    根据漂移测点索引列表自动生成漂移方案名称

    命名规则:
    - "all" -> "drift_all"（全通道温度漂移）
    - [2] -> "drift_2"
    - [2, 3, 4] -> "drift_2to4"
    - [1, 3, 5] -> "drift_1_3_5" (非连续)
    - None 或 [] -> "none"

    Args:
        drift_indices: 漂移测点索引，可以是 "all"（全通道）或索引列表

    Returns:
        漂移方案名称字符串
    """
    if drift_indices == "all":
        return "drift_all"

    if drift_indices is None or len(drift_indices) == 0:
        return "none"

    sorted_indices = sorted(drift_indices)

    if len(sorted_indices) == 1:
        return f"drift_{sorted_indices[0]}"

    # 检查是否连续
    is_continuous = all(
        sorted_indices[i+1] - sorted_indices[i] == 1
        for i in range(len(sorted_indices) - 1)
    )

    if is_continuous:
        return f"drift_{sorted_indices[0]}to{sorted_indices[-1]}"
    else:
        # 非连续情况，用下划线连接
        return "drift_" + "_".join(str(i) for i in sorted_indices)


def expand_case_drifts(case: dict) -> list:
    """
    展开case的drift配置

    如果drift是嵌套数组（如[[0],[1],[2]]），则展开为多个独立的case。
    如果drift是单个数组（如[0]）或无drift，则返回原始case。

    Args:
        case: case配置字典

    Returns:
        展开后的case列表
    """
    drift_value = case.get('drift')

    if not drift_value:
        return [case]

    if is_nested_drift(drift_value):
        expanded_cases = []
        for single_drift in drift_value:
            expanded_case = case.copy()
            expanded_case['drift'] = single_drift
            expanded_cases.append(expanded_case)
        return expanded_cases
    else:
        return [case]


def apply_drift_to_data(
    V_data: np.ndarray,
    drift_indices,
    drift_type: str = 'scale',
    drift_ratio: float = 0.1
) -> np.ndarray:
    """
    对数据应用温漂（读数漂移）

    Args:
        V_data: 原始数据矩阵，形状 (n_samples, n_features)
        drift_indices: 要应用漂移的测点索引，可以是：
            - "all": 全通道漂移（温度引起的同步漂移）
            - list: 指定通道索引列表
        drift_type: 漂移类型
            - 'scale': 比例漂移，drifted = original * (1 + drift_ratio)
            - 'offset': 绝对偏移，drifted = original + offset_value（需配合drift_ratio使用）
        drift_ratio: 漂移幅度
            - 对于'scale'类型：0.1表示10%的比例漂移
            - 对于'offset'类型：直接作为偏移值

    Returns:
        应用漂移后的数据矩阵（副本，不修改原数据）
    """
    # 全通道漂移（温度引起）
    if drift_indices == "all":
        if drift_type == 'scale':
            return V_data * (1 + drift_ratio)
        elif drift_type == 'offset':
            return V_data + V_data.mean(axis=0) * drift_ratio
        else:
            print(f"  [警告] 未知的漂移类型: {drift_type}，使用默认'scale'")
            return V_data * (1 + drift_ratio)

    # 创建数据副本
    V_drifted = V_data.copy()

    if drift_indices is None or len(drift_indices) == 0:
        return V_drifted

    # 确保索引在有效范围内
    valid_indices = [idx for idx in drift_indices if 0 <= idx < V_data.shape[1]]

    if len(valid_indices) != len(drift_indices):
        invalid = set(drift_indices) - set(valid_indices)
        print(f"  [警告] 部分漂移索引超出范围: {invalid}（总列数: {V_data.shape[1]}）")

    if not valid_indices:
        return V_drifted

    # 应用漂移
    if drift_type == 'scale':
        for idx in valid_indices:
            V_drifted[:, idx] = V_data[:, idx] * (1 + drift_ratio)
    elif drift_type == 'offset':
        for idx in valid_indices:
            col_mean = np.mean(V_data[:, idx])
            V_drifted[:, idx] = V_data[:, idx] + col_mean * drift_ratio
    else:
        print(f"  [警告] 未知的漂移类型: {drift_type}，使用默认'scale'")
        for idx in valid_indices:
            V_drifted[:, idx] = V_data[:, idx] * (1 + drift_ratio)

    return V_drifted


def extract_measures_suffix(measures_filename: str) -> str:
    """
    从测点文件名提取后缀标识

    例如：
        measures_ID_original.csv -> original
        measures_ID_offset_3.csv -> offset_3
        measures_ID.csv -> default

    Args:
        measures_filename: 测点文件名

    Returns:
        后缀字符串
    """
    # 移除扩展名
    name_without_ext = os.path.splitext(measures_filename)[0]

    # 尝试提取 measures_ID_ 后面的部分
    if 'measures_ID_' in name_without_ext:
        suffix = name_without_ext.split('measures_ID_')[1]
        return suffix if suffix else 'default'

    # 如果没有后缀，返回default
    return 'default'


def parse_dataset_name(dataset_name: str) -> tuple:
    """
    解析数据集名称，提取 FEM 模型和测点类型

    命名格式：{fem_model}_{measures_suffix}_{count} 或 {fem_model}
    例如：
        health_original_2000 -> ('health', 'original', 2000)
        health_offset_3_2000 -> ('health', 'offset_3', 2000)
        health_drift_0_2000 -> ('health', 'drift_0', 2000)
        first_damage_offset_3 -> ('first_damage', 'offset_3', None)
        health -> ('health', None, None)

    Args:
        dataset_name: 数据集名称

    Returns:
        (fem_model, measures_suffix, count) 元组
    """
    # 按长度降序排列，确保 damage_repaired_12 优先匹配于 damage_repaired
    known_models = sorted(KNOWN_MODELS, key=len, reverse=True)

    # 尝试匹配已知模型名称
    for model in known_models:
        if dataset_name.startswith(model):
            remaining = dataset_name[len(model):]
            if not remaining:
                return (model, None, None)

            # 去掉开头的下划线
            if remaining.startswith('_'):
                remaining = remaining[1:]
            else:
                continue  # 不是有效格式

            # 使用阈值区分 offset/drift 索引和 sample count
            # offset/drift 索引是小数字（通常 0-49），sample count 是大数字（>=50）
            COUNT_THRESHOLD = 50

            # 检查是否包含 offset 信息
            if remaining.startswith('offset'):
                # 解析 offset_X 或 offset_XtoY 格式
                parts = remaining.split('_')
                # 找到数字结尾（count）- 必须 >= COUNT_THRESHOLD 才算 count
                if parts[-1].isdigit() and int(parts[-1]) >= COUNT_THRESHOLD:
                    count = int(parts[-1])
                    measures_suffix = '_'.join(parts[:-1])
                else:
                    count = None
                    measures_suffix = remaining
                return (model, measures_suffix, count)
            # 检查是否包含 drift 信息
            elif remaining.startswith('drift'):
                # 解析 drift_X 或 drift_XtoY 格式
                parts = remaining.split('_')
                # 找到数字结尾（count）- 必须 >= COUNT_THRESHOLD 才算 count
                if parts[-1].isdigit() and int(parts[-1]) >= COUNT_THRESHOLD:
                    count = int(parts[-1])
                    measures_suffix = '_'.join(parts[:-1])
                else:
                    count = None
                    measures_suffix = remaining
                return (model, measures_suffix, count)
            elif remaining.startswith('original'):
                # original 或 original_2000 格式
                parts = remaining.split('_')
                if len(parts) >= 2 and parts[-1].isdigit() and int(parts[-1]) >= COUNT_THRESHOLD:
                    count = int(parts[-1])
                    measures_suffix = '_'.join(parts[:-1])
                else:
                    count = None
                    measures_suffix = 'original'
                return (model, measures_suffix, count)
            else:
                # 尝试解析为 measures_suffix_count 格式
                parts = remaining.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) >= COUNT_THRESHOLD:
                    return (model, parts[0], int(parts[1]))
                else:
                    return (model, remaining, None)

    # 未能匹配已知模型，返回原始名称
    return (dataset_name, None, None)


# ========================================
# 内嵌工具函数：绘图
# ========================================

def apply_plot_style():
    """应用学术风绘图样式"""
    plt.rcParams.update(PLOT_STYLE)


def save_figure(
    fig: plt.Figure,
    output_dir: str,
    name: str,
    data: dict[str, np.ndarray] | None = None,
):
    """保存图像和同名数据源"""
    os.makedirs(output_dir, exist_ok=True)
    img_path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(img_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    if data:
        df = pd.DataFrame(data)
        csv_path = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(csv_path, index=False)


def safe_kde(data: np.ndarray, grid: Optional[np.ndarray] = None) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """安全的核密度估计"""
    clean = data[np.isfinite(data)]
    if clean.size == 0:
        return None, None
    if clean.size == 1 or np.allclose(clean, clean[0]):
        center = clean[0]
        span = 1.0 if not np.isfinite(center) else max(abs(center) * 0.05, 1e-3)
        xs = np.linspace(center - span, center + span, 512)
        ys = np.exp(-0.5 * ((xs - center) / (span if span > 0 else 1e-3)) ** 2)
        ys /= np.trapezoid(ys, xs)
        return xs, ys

    if grid is None:
        low, high = np.percentile(clean, [0.5, 99.5])
        if np.isclose(low, high):
            delta = max(abs(low) * 0.05, 1e-3)
            low -= delta
            high += delta
        xs = np.linspace(low, high, 512)
    else:
        xs = grid

    std = clean.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        std = max(abs(clean.mean()) * 0.05, 1e-3)
    bandwidth = 1.06 * std * clean.size ** (-1 / 5)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = 1e-2

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(clean[:, None])
    log_density = kde.score_samples(xs[:, None])
    density = np.exp(log_density)
    density /= np.trapezoid(density, xs)
    return xs, density


def plot_grid_scatter_density(
    data: np.ndarray,
    title_list: list[str],
    output_dir: str,
    name: str,
    samples_per_row: int = 5,
    max_rows: int = 5,
    scatter_color: str = "tab:blue",
    density_color: str = "tab:green",
    scatter_size: float = 1.0,
    density_xlim: Optional[tuple[float, float]] = None,
    show_grid: bool = True,
    kde_linewidth: float = 1.0,
    progress_callback: Optional[Callable[[int, str], range]] = None,
) -> None:
    """标准化的网格子图绘制函数"""
    n_features = min(data.shape[1], len(title_list))
    cols = samples_per_row * 2  # 每个样本占2列（左散点，右密度）
    rows = max(1, min(max_rows, math.ceil(n_features / samples_per_row)))
    max_samples = rows * samples_per_row
    n = min(n_features, max_samples)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.35, rows * 1.35), sharey=False)
    axes = np.atleast_2d(axes).reshape(rows, cols)

    data_to_csv: dict[str, np.ndarray] = {"sample_idx": np.arange(data.shape[0])}
    global_scatter_ymin, global_scatter_ymax = np.inf, -np.inf
    global_density_ymax = 0.0
    scatter_axes: list[plt.Axes] = []
    density_axes: list[plt.Axes] = []

    # 进度迭代器
    if progress_callback:
        iter_range = progress_callback(rows * samples_per_row, "绘图进度")
    else:
        iter_range = range(rows * samples_per_row)

    # 绘制所有子图
    for idx in iter_range:
        sample_index = idx
        row = idx // samples_per_row
        col_sample = idx % samples_per_row
        scatter_ax = axes[row, col_sample * 2]
        density_ax = axes[row, col_sample * 2 + 1]

        if sample_index >= n:
            scatter_ax.axis("off")
            density_ax.axis("off")
            continue

        series = data[:, sample_index]
        sample_idx_arr = np.arange(series.shape[0])

        # 左边：绘制散点图
        scatter_ax.scatter(sample_idx_arr, series, s=scatter_size, color=scatter_color, alpha=0.7, linewidths=0)
        if show_grid:
            scatter_ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
        scatter_ax.tick_params(axis="both", which="both", direction="in", labelsize=8)
        scatter_ax.set_title(title_list[sample_index], fontsize=9, pad=2)

        # 收集散点图的Y轴范围
        clean = series[np.isfinite(series)]
        if clean.size:
            global_scatter_ymin = min(global_scatter_ymin, float(clean.min()))
            global_scatter_ymax = max(global_scatter_ymax, float(clean.max()))

        # 右边：绘制密度图（统计图）
        if clean.size == 0:
            density_ax.axis("off")
            continue

        grid_low, grid_high = np.percentile(clean, [0.5, 99.5])
        if np.isclose(grid_low, grid_high):
            spread = max(abs(grid_low) * 0.05, 1e-3)
            grid_low -= spread
            grid_high += spread
        grid = np.linspace(grid_low, grid_high, 512)

        xs, ys = safe_kde(series, grid)
        if xs is None:
            density_ax.axis("off")
            continue

        hist_vals, _, _ = density_ax.hist(
            series,
            bins=PREPROCESS_HIST_BINS,
            density=True,
            color=density_color,
            alpha=PREPROCESS_HIST_ALPHA,
            edgecolor="black",
            linewidth=0.5,
        )
        density_ax.fill_between(xs, ys, color=density_color, alpha=PREPROCESS_HIST_ALPHA)
        density_ax.plot(xs, ys, color=density_color, lw=kde_linewidth)
        density_ax.axhline(0, color="black", lw=0.5, alpha=0.4)
        if show_grid:
            density_ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
        density_ax.tick_params(axis="both", which="both", direction="in", labelsize=8)

        ymax_local = max(
            (hist_vals.max() if hist_vals.size else 0.0), (ys.max() if ys is not None else 0.0)
        )
        global_density_ymax = max(global_density_ymax, ymax_local)

        scatter_axes.append(scatter_ax)
        density_axes.append(density_ax)
        data_to_csv[f"feature_{sample_index}"] = series

    # 统一设置所有散点图的Y轴范围
    if np.isfinite(global_scatter_ymin) and np.isfinite(global_scatter_ymax) and global_scatter_ymin < global_scatter_ymax:
        pad = 0.02 * (global_scatter_ymax - global_scatter_ymin)
        for ax in scatter_axes:
            ax.set_ylim(global_scatter_ymin - pad, global_scatter_ymax + pad)
            ax.set_xlim(0, max(0, data.shape[0] - 1))

    # 统一设置所有密度图的Y轴范围
    if global_density_ymax > 0:
        for ax in density_axes:
            ax.set_ylim(0, global_density_ymax * 1.05)

    # 如果指定了密度图的X轴范围
    if density_xlim is not None:
        for ax in density_axes:
            ax.set_xlim(density_xlim)

    # 隐藏除了第一行第一列外的所有刻度标签
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if not (r == 0 and c in (0, 1)):
                ax.set_xticklabels([])
                ax.set_yticklabels([])

    fig.tight_layout(pad=0.6)
    fig.subplots_adjust(hspace=0.25, wspace=0.2)
    save_figure(fig, output_dir, name=name, data=data_to_csv)


# ========================================
# 数据收集函数
# ========================================

def collect_data(
    source_dir: str,
    max_folders: int,
    v_ids: list[int],
    use_npy: bool = True
) -> np.ndarray:
    """
    收集单个数据集的数据

    Args:
        source_dir: 数据源目录
        max_folders: 最大处理文件夹数
        v_ids: 测点ID列表
        use_npy: 是否使用NPY格式

    Returns:
        V_data: 形状为(n_samples, n_features)的numpy数组
    """
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"错误: 数据源目录不存在: {source_dir}")

    # 获取文件夹列表
    folders = []
    for f in os.listdir(source_dir):
        fp = os.path.join(source_dir, f)
        if not os.path.isdir(fp):
            continue
        if f.isdigit():
            folders.append(f)
            continue
        # 兼容非数字命名
        try:
            has_npy = any(x.lower().endswith('.npy') for x in os.listdir(fp))
            has_csv = any(x.lower().endswith('.csv') for x in os.listdir(fp))
            if (use_npy and has_npy) or ((not use_npy) and has_csv):
                folders.append(f)
        except Exception:
            continue

    folders = sorted(folders, key=lambda x: int(x) if x.isdigit() else 0)
    total_folders = len(folders)
    if max_folders is not None:
        folders = folders[:max_folders]

    print(f"  找到 {total_folders} 个文件夹, 处理前 {len(folders)} 个")

    num_points = len(v_ids)
    total_samples = len(folders)
    V_data = np.zeros((total_samples, num_points))

    sample_idx = 0
    skipped_folders = []
    required_ids_set = set(v_ids)

    for folder in tqdm(folders, desc="  收集进度"):
        try:
            folder_path = os.path.join(source_dir, folder)
            expected_file = "iteration.npy" if use_npy else "iteration.csv"
            file_path = os.path.join(folder_path, expected_file)

            if not os.path.exists(file_path):
                ext = ".npy" if use_npy else ".csv"
                files = [f for f in os.listdir(folder_path) if f.lower().endswith(ext)]
                if len(files) == 1:
                    file_path = os.path.join(folder_path, files[0])
                else:
                    skipped_folders.append(folder)
                    continue

            if use_npy:
                arr = np.load(file_path)
                cols = (
                    ["Element Label", "S-Mises"]
                    if arr.shape[1] == 2
                    else ["Element Label", "S-Mises", "X", "Y", "Z"][: arr.shape[1]]
                )
                df = pd.DataFrame(arr, columns=cols).set_index("Element Label")
            else:
                df = pd.read_csv(file_path, skipinitialspace=True).set_index("Element Label")

            actual_labels_set = set(df.index)
            missing_required_ids = required_ids_set - actual_labels_set
            if missing_required_ids:
                raise ValueError(f"缺少必要的ID: {sorted(list(missing_required_ids))}")

            V_data[sample_idx] = df.loc[v_ids]["S-Mises"].values
            sample_idx += 1

        except Exception:
            skipped_folders.append(folder)
            continue

    if sample_idx < total_samples:
        print(f"  成功处理 {sample_idx} / {len(folders)} 个文件夹")
        V_data = V_data[:sample_idx]

    if skipped_folders:
        print(f"  跳过 {len(skipped_folders)} 个文件夹")

    return V_data


# ========================================
# 可视化函数
# ========================================

def visualize_dataset(
    V_data: np.ndarray,
    output_dir: str,
    dataset_name: str,
    v_ids: list[int]
):
    """
    可视化单个数据集

    Args:
        V_data: 数据矩阵
        output_dir: 输出目录
        dataset_name: 数据集名称
        v_ids: 测点ID列表
    """
    apply_plot_style()

    def _progress_iter(count: int, desc: str):
        return tqdm(range(count), desc=desc) if SHOW_PLOT_PROGRESS else range(count)

    # 准备标题列表
    title_list = [f"V[{i}]  ID {v_ids[i]}" for i in range(len(v_ids))]

    # 绘制原始数据
    print(f"  绘制可视化（前{min(PREPROCESS_PLOT_FIRST_N, V_data.shape[1])}个维度）")
    plot_grid_scatter_density(
        data=V_data,
        title_list=title_list,
        output_dir=output_dir,
        name=f"distribution_grid_{min(PREPROCESS_PLOT_FIRST_N, V_data.shape[1])}dims",
        samples_per_row=PREPROCESS_GRID_COLS,
        max_rows=PREPROCESS_GRID_ROWS,
        scatter_color=PREPROCESS_SCATTER_COLOR,
        density_color=PREPROCESS_DENSITY_COLOR,
        scatter_size=PREPROCESS_SCATTER_SIZE,
        density_xlim=None,  # 原始数据自动范围
        show_grid=PREPROCESS_SHOW_GRID,
        kde_linewidth=PREPROCESS_KDE_LINEWIDTH,
        progress_callback=_progress_iter if SHOW_PLOT_PROGRESS else None,
    )


# ========================================
# 单数据集预处理函数
# ========================================

def preprocess_dataset(dataset_name: str, config: dict, v_ids: list[int]):
    """
    预处理单个数据集

    Args:
        dataset_name: 数据集名称
        config: 数据集配置字典
        v_ids: 测点ID列表
    """
    print("\n" + "=" * 60)
    print(f"处理数据集: {config['name']} ({dataset_name})")
    print("=" * 60)
    print(f"  源目录: {config['source_dir']}")
    print(f"  最大样本数: {config['max_samples']}")

    # 1. 收集数据
    print("\n[步骤1] 数据收集")
    V_data = collect_data(
        source_dir=config['source_dir'],
        max_folders=config['max_samples'],
        v_ids=v_ids,
        use_npy=USE_NPY
    )

    # 2. 保存原始数据
    output_path = os.path.join(OUTPUT_DIR, config['output_subdir'])
    os.makedirs(output_path, exist_ok=True)

    npz_path = os.path.join(output_path, 'preprocessed_data_raw.npz')
    np.savez(npz_path, V=V_data)
    print(f"\n[步骤2] 保存数据")
    print(f"  已保存到: {npz_path}")
    print(f"  数据形状: {V_data.shape}")

    # 3. 可视化统计
    if DO_VISUALIZE:
        print(f"\n[步骤3] 可视化")
        visualize_dataset(
            V_data=V_data,
            output_dir=output_path,
            dataset_name=config['name'],
            v_ids=v_ids
        )
        print(f"  可视化已保存到: {output_path}")

    print(f"\n数据集 {dataset_name} 处理完成！")
    return V_data


# ========================================
# 主程序
# ========================================

def main():
    """主流程：依次处理所有数据集"""
    print("=" * 60)
    print("流程04：统一数据预处理（所有数据集）")
    print("=" * 60)
    print(f"输出根目录: {OUTPUT_DIR}")
    print(f"测点ID文件: {MEASURES_ID_CSV}")
    print(f"数据收集: 是")
    print(f"可视化: {'是' if DO_VISUALIZE else '否'}")
    print("=" * 60)

    # 初始化随机种子
    random.seed(SEED)
    np.random.seed(SEED)

    # ========================================
    # 基于Case配置列表处理数据（声明式配置）
    # ========================================
    if not CASE_CONFIGS:
        print("\n错误: CASE_CONFIGS为空,没有配置需要处理的case")
        return

    print("\n" + "=" * 60)
    print("基于Case配置列表处理数据集")
    print("=" * 60)

    # 统计配置信息
    enabled_cases = [c for c in CASE_CONFIGS if c.get('enabled', False)]
    print(f"总配置数: {len(CASE_CONFIGS)}, 启用数: {len(enabled_cases)}")

    # 用于记录处理结果
    case_results = {}

    for case_idx, case_config in enumerate(CASE_CONFIGS):
        if not case_config.get('enabled', False):
            print(f"\n[Case {case_idx + 1}] 跳过（未启用）: {case_config}")
            continue

        # 提取配置参数
        fem_structure = case_config['fem_structure']
        measures_file = case_config['measures_file']
        sample_count = case_config['sample_count']

        # 提取测点文件后缀（用于查找CSV中的列名）
        measures_suffix = extract_measures_suffix(measures_file)

        # 生成输出文件夹名称
        # 优先使用配置中的 output_folder（支持自动生成的偏移名称如 offset_2, offset_2to4）
        # 否则回退到旧逻辑：FEM结构_测点文件后缀_生成数量
        if 'output_folder' in case_config:
            output_folder_name = case_config['output_folder']
        else:
            output_folder_name = f"{fem_structure}_{measures_suffix}_{sample_count}"

        print("\n" + "=" * 60)
        print(f"[Case {case_idx + 1}] 处理数据集: {output_folder_name}")
        print("=" * 60)
        print(f"  FEM结构: {fem_structure}")
        print(f"  测点文件: {measures_file}")
        print(f"  样本数量: {sample_count}")

        # 检查是否已存在完整输出
        output_path_case = os.path.join(OUTPUT_DIR, output_folder_name)
        _npz_path = os.path.join(output_path_case, 'preprocessed_data_raw.npz')
        if os.path.exists(_npz_path) and os.path.getsize(_npz_path) > 1024:
            print(f"  [SKIP] {output_folder_name} already exists")
            case_results[output_folder_name] = {
                'success': True,
                'skipped': True,
                'output_path': output_path_case
            }
            continue

        # 获取FEM结构配置
        dataset_config = DATASET_CONFIGS.get(fem_structure)
        if not dataset_config:
            print(f"\n警告: 未找到FEM结构配置 '{fem_structure}'，跳过此case")
            case_results[output_folder_name] = {'success': False, 'error': f'FEM结构配置未找到: {fem_structure}'}
            continue

        # 读取测点ID文件
        measures_csv_path = os.path.join(
            os.path.dirname(__file__),
            'AC_convert_and_extract_output',
            measures_file
        )
        if not os.path.exists(measures_csv_path):
            print(f"\n警告: 测点ID文件未找到: {measures_csv_path}")
            print("请先执行流程AC以生成此文件，跳过此case")
            case_results[output_folder_name] = {'success': False, 'error': f'测点ID文件未找到: {measures_csv_path}'}
            continue

        # 读取测点ID
        try:
            measures_df_case = pd.read_csv(measures_csv_path)

            # 尝试查找有效的测点ID列
            # 偏移测点文件使用 'offset_measures' 列，auto文件使用 'all_measures' 列
            case_v_col = None
            candidate_cols = [
                'offset_measures',  # 偏移测点文件的固定列名
                measures_suffix + '_measures',  # 如 auto_measures
                'all_measures',
                RECONSTRUCT_COLUMN
            ]
            for col in candidate_cols:
                if col in measures_df_case.columns:
                    case_v_col = col
                    break

            if case_v_col is None:
                print(f"\n警告: 测点CSV文件中未找到有效列")
                print(f"可用列: {list(measures_df_case.columns)}")
                case_results[output_folder_name] = {'success': False, 'error': '测点CSV无有效列'}
                continue

            case_v_ids = measures_df_case[case_v_col].dropna().astype(int).tolist()
            print(f"  测点维度: {len(case_v_ids)} (列 '{case_v_col}')")

            # 收集数据
            V_case_data = collect_data(
                source_dir=dataset_config['source_dir'],
                max_folders=sample_count,
                v_ids=case_v_ids,
                use_npy=USE_NPY
            )

            # 检查是否需要应用温漂
            drift_indices = case_config.get('drift_indices')
            drift_type = case_config.get('drift_type', 'scale')
            drift_ratio = case_config.get('drift_ratio', 0.1)

            if drift_indices:
                print(f"\n  [温漂处理] 应用温漂: {drift_indices}")
                print(f"    漂移类型: {drift_type}, 漂移幅度: {drift_ratio}")

                # 记录漂移前的统计信息
                if drift_indices == "all":
                    orig_mean = np.mean(V_case_data)
                    orig_std = np.std(V_case_data)
                    print(f"    漂移前 - 全通道: mean={orig_mean:.4f}, std={orig_std:.4f}")
                else:
                    for idx in drift_indices:
                        if 0 <= idx < V_case_data.shape[1]:
                            orig_mean = np.mean(V_case_data[:, idx])
                            orig_std = np.std(V_case_data[:, idx])
                            print(f"    漂移前 - 列[{idx}]: mean={orig_mean:.4f}, std={orig_std:.4f}")

                # 应用温漂
                V_case_data = apply_drift_to_data(
                    V_case_data,
                    drift_indices,
                    drift_type,
                    drift_ratio
                )

                # 记录漂移后的统计信息
                if drift_indices == "all":
                    new_mean = np.mean(V_case_data)
                    new_std = np.std(V_case_data)
                    print(f"    漂移后 - 全通道: mean={new_mean:.4f}, std={new_std:.4f}")
                else:
                    for idx in drift_indices:
                        if 0 <= idx < V_case_data.shape[1]:
                            new_mean = np.mean(V_case_data[:, idx])
                            new_std = np.std(V_case_data[:, idx])
                            print(f"    漂移后 - 列[{idx}]: mean={new_mean:.4f}, std={new_std:.4f}")

            # 保存数据
            output_path_case = os.path.join(OUTPUT_DIR, output_folder_name)
            os.makedirs(output_path_case, exist_ok=True)
            npz_path = os.path.join(output_path_case, 'preprocessed_data_raw.npz')
            np.savez(npz_path, V=V_case_data)
            print(f"  已保存到: {npz_path}")
            print(f"  数据形状: {V_case_data.shape}")

            # 可视化
            if DO_VISUALIZE:
                print(f"  绘制可视化...")
                visualize_dataset(
                    V_data=V_case_data,
                    output_dir=output_path_case,
                    dataset_name=output_folder_name,
                    v_ids=case_v_ids
                )
                print(f"  可视化已保存")

            print(f"  数据集 {output_folder_name} 处理完成！")
            case_results[output_folder_name] = {
                'success': True,
                'shape': V_case_data.shape,
                'output_path': output_path_case
            }

        except Exception as e:
            print(f"\n警告: Case {output_folder_name} 处理失败: {e}")
            import traceback
            traceback.print_exc()
            case_results[output_folder_name] = {'success': False, 'error': str(e)}

    # 输出Case处理汇总
    print("\n" + "=" * 60)
    print("Case配置处理完成！结果汇总：")
    print("=" * 60)

    for case_name, result in case_results.items():
        if result.get('skipped'):
            print(f"[SKIP] {case_name:40s} - {result['output_path']}")
        elif result['success']:
            print(f"[OK] {case_name:40s} - {result['shape']} - {result['output_path']}")
        else:
            print(f"[FAIL] {case_name:40s} - 失败: {result['error']}")

    print("=" * 60)


# ========================================
# External Config Interface (for AAA_oneclick_run.py)
# ========================================

def collect_all_offsets(config: dict) -> list:
    """
    收集所有case中配置的offset方案

    支持两种offset配置格式：
    - 单个方案: [0] 或 [1,2,3]
    - 批量方案: [[0],[1],[2],[3],[1,2],[3,4]]

    Args:
        config: 全局配置字典

    Returns:
        去重后的offset方案列表，每个元素是一个offset索引列表
        例如: [[0], [1], [2], [3], [1,2], [3,4]]
    """
    cases = config.get('cases', [])
    global_offset = config.get('offset')  # 兼容旧的全局offset配置

    offsets = []
    seen = set()

    # 收集case级别的offset
    for case in cases:
        offset = case.get('offset')
        if offset:
            if is_nested_offset(offset):
                # 批量offset方案，展开每个子方案
                for single_offset in offset:
                    key = tuple(sorted(single_offset))
                    if key not in seen:
                        seen.add(key)
                        offsets.append(single_offset)
            else:
                # 单个offset方案
                key = tuple(sorted(offset))
                if key not in seen:
                    seen.add(key)
                    offsets.append(offset)

    # 兼容：如果有全局offset且没有case级别offset，使用全局offset
    if global_offset and not offsets:
        if is_nested_offset(global_offset):
            offsets.extend(global_offset)
        else:
            offsets.append(global_offset)

    return offsets


def collect_all_drifts(config: dict) -> list:
    """
    收集所有case中配置的drift方案

    支持两种drift配置格式：
    - 单个方案: [0] 或 [1,2,3]
    - 批量方案: [[0],[1],[2],[3],[1,2],[3,4]]

    Args:
        config: 全局配置字典

    Returns:
        去重后的drift方案列表，每个元素是一个包含drift配置的字典
        例如: [{'indices': [0], 'type': 'scale', 'ratio': 0.1}, ...]
    """
    cases = config.get('cases', [])

    drifts = []
    seen = set()

    # 收集case级别的drift
    for case in cases:
        drift = case.get('drift')
        if drift:
            drift_type = case.get('drift_type', 'scale')
            drift_ratio = case.get('drift_ratio', 0.1)

            if is_nested_drift(drift):
                # 批量drift方案，展开每个子方案
                for single_drift in drift:
                    key = (tuple(sorted(single_drift)), drift_type, drift_ratio)
                    if key not in seen:
                        seen.add(key)
                        drifts.append({
                            'indices': single_drift,
                            'type': drift_type,
                            'ratio': drift_ratio
                        })
            else:
                # 单个drift方案（包括 "all" 全通道）
                if drift == "all":
                    key = ("all", drift_type, drift_ratio)
                else:
                    key = (tuple(sorted(drift)), drift_type, drift_ratio)
                if key not in seen:
                    seen.add(key)
                    drifts.append({
                        'indices': drift,
                        'type': drift_type,
                        'ratio': drift_ratio
                    })

    return drifts


def resolve_case_datasets(case: dict, config: dict) -> dict:
    """
    解析case的数据集配置，处理offset和drift字段

    如果case配置了offset字段：
    - old_baseline: 使用原始非偏移测点数据（如 health -> health_original_2000）
    - new_baseline: 使用偏移测点数据（如 health + offset=[2,6,9] -> health_offset_2to9_500）
    - damage_test: 使用偏移测点数据（如 first_damage + offset=[2,6,9] -> first_damage_offset_2to9_100）

    如果case配置了drift字段：
    - old_baseline: 使用原始非漂移数据（如 health -> health_original_2000）
    - new_baseline: 使用漂移数据（如 health + drift=[0] -> health_drift_0_2000）
    - damage_test: 使用漂移数据（如 first_damage + drift=[0] -> first_damage_drift_0_100）

    Args:
        case: case配置字典
        config: 全局配置字典

    Returns:
        解析后的数据集配置字典 {
            'old_baseline': 解析后的完整数据集名称,
            'new_baseline': 解析后的完整数据集名称,
            'damage_test': 解析后的完整数据集名称,
            'offset': offset配置（如果有）,
            'drift': drift配置（如果有）,
            'drift_type': 漂移类型（如果有）,
            'drift_ratio': 漂移幅度（如果有）
        }
    """
    simulation_counts = config.get('simulation_counts', {})
    offset_indices = case.get('offset')  # case级别的offset配置
    drift_indices = case.get('drift')    # case级别的drift配置
    drift_type = case.get('drift_type', 'scale')
    drift_ratio = case.get('drift_ratio', 0.1)

    result = {
        'offset': offset_indices,
        'drift': drift_indices,
        'drift_type': drift_type,
        'drift_ratio': drift_ratio
    }

    # 处理 old_baseline（始终使用原始非偏移/非漂移数据）
    old_baseline = case.get('old_baseline')
    if old_baseline:
        fem_model, measures_suffix, _ = parse_dataset_name(old_baseline)
        # old_baseline 始终使用 original
        if measures_suffix is None or (offset_indices and not measures_suffix.startswith('offset')) or (drift_indices and not measures_suffix.startswith('drift')):
            measures_suffix = 'original'
        count = simulation_counts.get(fem_model, 0)
        result['old_baseline'] = f"{fem_model}_{measures_suffix}_{count}" if count else old_baseline

    # 处理 new_baseline 和 damage_test
    for role in ['new_baseline', 'damage_test']:
        dataset_name = case.get(role)
        if not dataset_name:
            continue

        fem_model, measures_suffix, _ = parse_dataset_name(dataset_name)
        count = simulation_counts.get(fem_model, 0)

        if offset_indices:
            # 如果case配置了offset，new_baseline和damage_test使用偏移数据
            offset_name = get_offset_name(offset_indices)
            result[role] = f"{fem_model}_{offset_name}_{count}" if count else f"{fem_model}_{offset_name}"
        elif drift_indices:
            # 如果case配置了drift，new_baseline和damage_test使用漂移数据
            drift_name = get_drift_name(drift_indices)
            result[role] = f"{fem_model}_{drift_name}_{count}" if count else f"{fem_model}_{drift_name}"
        else:
            # 没有offset/drift配置，使用数据集名称本身的配置
            if measures_suffix is None:
                measures_suffix = 'original'
            result[role] = f"{fem_model}_{measures_suffix}_{count}" if count else dataset_name

    return result


def run_from_config(config: dict) -> int:
    """
    从外部配置运行本脚本

    解析TL_settings.jsonc中的:
    - FEM_models: 模型列表
    - simulation_counts: 各模型的仿真数据量
    - cases: 场景配置，支持case级别的offset和drift字段
    - offset: 全局偏移测点行索引列表（兼容旧配置）

    数据集命名规则:
    - 普通case: {fem_model}_{measures_suffix}_{count}
    - offset case: old_baseline使用original，new_baseline和damage_test使用offset_{...}
    - drift case: old_baseline使用original，new_baseline和damage_test使用drift_{...}

    Args:
        config: 从TL_settings.jsonc加载的配置字典

    Returns:
        0 表示成功，非0 表示失败
    """
    global CASE_CONFIGS, DATASET_CONFIGS, KNOWN_MODELS

    print("\n" + "=" * 70)
    print("[AD] 从外部配置运行数据预处理")
    print("=" * 70)

    # 1. 解析配置
    fem_models = config.get('FEM_models', {})
    simulation_counts = config.get('simulation_counts', {})

    # 动态扩展已知模型名称（支持 damage_repaired_12 等带参数的模型名）
    for model_name in fem_models.keys():
        if model_name not in KNOWN_MODELS:
            KNOWN_MODELS.append(model_name)
    for model_name in simulation_counts.keys():
        if model_name not in KNOWN_MODELS:
            KNOWN_MODELS.append(model_name)

    # 展开批量offset和drift后的cases
    expanded_cases = expand_all_cases(config)
    original_count = len(config.get('cases', []))
    print(f"[配置] 展开cases: {original_count} -> {len(expanded_cases)} 个")

    # 收集所有offset方案
    all_offsets = collect_all_offsets(config)
    print(f"[配置] 收集到 {len(all_offsets)} 个offset方案:")
    for offset_indices in all_offsets:
        offset_name = get_offset_name(offset_indices)
        print(f"  - {offset_indices} -> {offset_name}")

    # 收集所有drift方案
    all_drifts = collect_all_drifts(config)
    print(f"[配置] 收集到 {len(all_drifts)} 个drift方案:")
    for drift_config in all_drifts:
        drift_name = get_drift_name(drift_config['indices'])
        print(f"  - {drift_config['indices']} -> {drift_name} (type={drift_config['type']}, ratio={drift_config['ratio']})")

    # 2. 收集需要生成的数据集（使用展开后的cases和resolve_case_datasets解析）
    needed_datasets = set()
    dataset_to_offset = {}  # 记录数据集对应的offset配置
    dataset_to_drift = {}   # 记录数据集对应的drift配置

    for case in expanded_cases:
        resolved = resolve_case_datasets(case, config)
        offset_indices = case.get('offset')
        drift_indices = case.get('drift')
        drift_type = case.get('drift_type', 'scale')
        drift_ratio = case.get('drift_ratio', 0.1)

        for role in ['old_baseline', 'new_baseline', 'damage_test']:
            dataset_dir = resolved.get(role)
            if dataset_dir:
                needed_datasets.add(dataset_dir)
                # 记录数据集对应的offset配置（用于后续确定测点文件）
                if offset_indices and role in ['new_baseline', 'damage_test']:
                    dataset_to_offset[dataset_dir] = offset_indices
                # 记录数据集对应的drift配置（用于后续应用漂移）
                if drift_indices and role in ['new_baseline', 'damage_test']:
                    dataset_to_drift[dataset_dir] = {
                        'indices': drift_indices,
                        'type': drift_type,
                        'ratio': drift_ratio
                    }

    print(f"\n[配置] 需要生成的数据集: {len(needed_datasets)} 个")
    for ds in sorted(needed_datasets):
        offset = dataset_to_offset.get(ds)
        drift = dataset_to_drift.get(ds)
        if offset:
            print(f"  - {ds} (offset: {offset})")
        elif drift:
            print(f"  - {ds} (drift: {drift['indices']}, type={drift['type']}, ratio={drift['ratio']})")
        else:
            print(f"  - {ds}")

    # 3. 转换为CASE_CONFIGS格式
    converted_case_configs = []

    for dataset_dir in needed_datasets:
        # 解析数据集目录名，提取 FEM 模型、测点类型和数量
        # 格式: {fem_model}_{measures_suffix}_{count}
        parts = dataset_dir.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            count = int(parts[1])
            model_and_suffix = parts[0]
        else:
            # 无法解析，跳过
            print(f"[跳过] 无法解析数据集目录名: {dataset_dir}")
            continue

        # 进一步解析 model_and_suffix
        fem_model, measures_suffix, _ = parse_dataset_name(model_and_suffix)
        if measures_suffix is None:
            measures_suffix = 'original'

        # 确定测点文件名
        if measures_suffix == 'original':
            measures_file = 'measures_ID_original.csv'
        elif measures_suffix.startswith('offset'):
            # 从 dataset_to_offset 获取对应的offset配置
            offset_indices = dataset_to_offset.get(dataset_dir)
            if offset_indices:
                measures_file = get_offset_measures_filename(offset_indices)
            else:
                # 回退：从measures_suffix解析offset
                measures_file = f'measures_ID_{measures_suffix}.csv'
        elif measures_suffix.startswith('drift'):
            # drift数据集使用原始测点文件，漂移在数据收集后应用
            measures_file = 'measures_ID_original.csv'
        else:
            measures_file = f'measures_ID_{measures_suffix}.csv'

        # 构建配置项
        case_config = {
            'fem_structure': fem_model,
            'measures_file': measures_file,
            'sample_count': count,
            'enabled': True,
            'output_folder': dataset_dir
        }

        # 如果是drift数据集，记录漂移配置
        drift_config = dataset_to_drift.get(dataset_dir)
        if drift_config:
            case_config['drift_indices'] = drift_config['indices']
            case_config['drift_type'] = drift_config['type']
            case_config['drift_ratio'] = drift_config['ratio']

        converted_case_configs.append(case_config)

        if drift_config:
            print(f"  - {dataset_dir}: {fem_model} + {measures_file} + {count}样本 (drift: {drift_config['indices']})")
        else:
            print(f"  - {dataset_dir}: {fem_model} + {measures_file} + {count}样本")

    # 4. 更新DATASET_CONFIGS（模型源目录映射）
    for model_name in fem_models.keys():
        if model_name not in DATASET_CONFIGS:
            # 添加新模型的源目录配置
            name_map = {
                'health': '健康状态',
                'first_damage': '初始损伤',
                'damage_repaired': '损伤修复',
                'second_damage': '二次损伤'
            }
            DATASET_CONFIGS[model_name] = {
                'name': name_map.get(model_name, model_name),
                'source_dir': rf'C:\SHM_abaqus_data\{model_name}',
                'max_samples': simulation_counts.get(model_name, 0),
                'output_subdir': model_name
            }

    # 5. 更新全局CASE_CONFIGS
    if converted_case_configs:
        CASE_CONFIGS.clear()
        CASE_CONFIGS.extend(converted_case_configs)
        print(f"\n[配置] 已更新CASE_CONFIGS: {len(CASE_CONFIGS)} 个配置")

    # 6. 显示配置摘要
    print("\n[配置摘要]")
    print(f"  数据集组合数: {len(converted_case_configs)}")
    for cfg in converted_case_configs:
        folder_name = cfg.get('output_folder', f"{cfg['fem_structure']}_{cfg['sample_count']}")
        print(f"    - {folder_name}")

    # 7. 调用原有的main函数逻辑
    print("\n[执行] 开始数据预处理...")
    try:
        main()
        return 0
    except Exception as e:
        print(f"[错误] 数据预处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


def load_jsonc(filepath: str) -> dict:
    """加载JSONC文件（支持//注释和尾随逗号）"""
    import re
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    # 去除 // 注释
    lines = content.split('\n')
    cleaned_lines = []
    for line in lines:
        in_string = False
        result = []
        i = 0
        while i < len(line):
            char = line[i]
            if char == '"' and (i == 0 or line[i-1] != '\\'):
                in_string = not in_string
                result.append(char)
            elif char == '/' and i + 1 < len(line) and line[i+1] == '/' and not in_string:
                break
            else:
                result.append(char)
            i += 1
        cleaned_lines.append(''.join(result))
    content = '\n'.join(cleaned_lines)
    content = re.sub(r',\s*([\]}])', r'\1', content)
    return json.loads(content)


if __name__ == "__main__":
    # 独立运行时，从 TL_settings.jsonc 加载配置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "TL_settings.jsonc")

    if os.path.exists(config_path):
        print(f"[配置] 从 {config_path} 加载配置")
        config = load_jsonc(config_path)
        sys.exit(run_from_config(config))
    else:
        print(f"[错误] 配置文件不存在: {config_path}")
        print("[提示] 请确保 TL_settings.jsonc 存在，或通过 AAA_oneclick_run.py 运行")
        sys.exit(1)
