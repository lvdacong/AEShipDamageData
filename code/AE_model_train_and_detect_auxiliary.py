"""
AE_model_train_and_detect_auxiliary.py
=======================================
Detection and visualization auxiliary: anomaly scores, residual
analysis, ROC metrics, 3D rendering, and comparison figure assembly.
Imported by AE stage scripts; not executed directly.
"""

import os
import re
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from scipy import stats
from scipy.ndimage import uniform_filter1d

# Visualization imports
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

# PyVista imports for 3D rendering
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

# PIL import for image merging
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ========================================
# 配置参数
# ========================================

# 可视化风格
VIS_STYLE = {
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
}
VIS_DPI = 300
N_SAMPLE_VISUALIZE = 5  # 可视化样本数

# 3D渲染配置
RENDER_WINDOW_WIDTH = 1920
RENDER_WINDOW_HEIGHT = 1080
RENDER_BACKGROUND = "white"
FEATURE_EDGE_ANGLE = 30
FEATURE_EDGE_COLOR = "black"
FEATURE_EDGE_WIDTH = 1.5

# 颜色映射参数
COLORMAP_NAME = "turbo"
USE_DISCRETE_COLORMAP = False
DISCRETE_COLORMAP_BANDS = 24
COLORMAP_RANGE = (0.05, 1.0)

# 插值方法
# Inverse Quadratic 核 φ(r) = 1/(1+(εr)²)：局部支持、平滑、无奇点、无伪影。
# ε=5e-4 → 特征影响半径 ~2000mm（远小于 252 测点的平均间距 ~5000mm），
# 使主峰保持局部化但有自然过渡，避免 TPS 的全局基线抬升。
INTERPOLATION_METHOD = "rbf"  # "rbf"(推荐) / "linear" / "nearest"
RBF_KERNEL = "inverse_quadratic"
RBF_SMOOTHING = 0.1
RBF_EPSILON = 0.0005

# 正态性检验
NORMALITY_TEST_ALPHA = 0.05


# ========================================
# 工具函数
# ========================================

def apply_vis_style():
    """应用学术风格"""
    plt.rcParams.update(VIS_STYLE)


# ========================================
# 损伤检测核心函数
# ========================================

def compute_residuals(V: np.ndarray, model: nn.Module, device: torch.device, batch_size: int = 512) -> np.ndarray:
    """
    计算残差（重构值 - 原始值）

    Args:
        V: (N, D) 输入数据
        model: 自编码器模型
        device: 设备
        batch_size: 批次大小

    Returns:
        残差矩阵 (N, D)
    """
    X = torch.from_numpy(V).to(device)
    all_residuals = []
    model.eval()
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            batch = X[i : i + batch_size]
            residuals = (model(batch) - batch).cpu().numpy()
            all_residuals.append(residuals)
    return np.concatenate(all_residuals, axis=0)


def compute_normal_params(residuals: np.ndarray) -> dict:
    """
    计算每个维度的正态分布参数（均值和标准差）

    Args:
        residuals: (N, D) 残差矩阵

    Returns:
        dict with:
            - means: (D,) 每维度均值
            - stds: (D,) 每维度标准差（去均值后）
            - residuals_centered: (N, D) 去均值后的残差
    """
    N, D = residuals.shape

    # 计算每维度均值
    means = residuals.mean(axis=0)

    # 去均值化
    residuals_centered = residuals - means

    # 计算标准差（使用ddof=1进行无偏估计）
    stds = residuals_centered.std(axis=0, ddof=1)

    # 处理边界情况：标准差为0或非有限值
    invalid_mask = ~np.isfinite(stds) | (stds == 0)
    if invalid_mask.any():
        print(f"  [警告] {invalid_mask.sum()}/{D} 个维度的标准差无效，设为1e-6")
        stds[invalid_mask] = 1e-6

    return {
        'means': means,
        'stds': stds,
        'residuals_centered': residuals_centered
    }


def test_normality(residuals_centered: np.ndarray, alpha: float = NORMALITY_TEST_ALPHA) -> dict:
    """
    对每个维度进行Shapiro-Wilk正态性检验

    Args:
        residuals_centered: (N, D) 去均值后的残差
        alpha: 显著性水平（默认0.05）

    Returns:
        dict with:
            - statistics: (D,) Shapiro-Wilk W统计量
            - p_values: (D,) p值
            - pass_rate: float, 通过检验的维度比例（p > α）
            - pass_mask: (D,) bool数组，True表示通过检验
    """
    N, D = residuals_centered.shape
    statistics = np.zeros(D)
    p_values = np.zeros(D)

    for dim in range(D):
        data = residuals_centered[:, dim]
        clean = data[np.isfinite(data)]

        if clean.size < 3:
            # 样本量太小，无法检验
            statistics[dim] = np.nan
            p_values[dim] = np.nan
            continue

        try:
            # Shapiro-Wilk检验（适用于样本量3-5000）
            if clean.size > 5000:
                # 对于大样本，抽样进行检验
                rng = np.random.default_rng(42)
                sample = rng.choice(clean, size=5000, replace=False)
                w_stat, p_val = stats.shapiro(sample)
            else:
                w_stat, p_val = stats.shapiro(clean)

            statistics[dim] = w_stat
            p_values[dim] = p_val

        except Exception as e:
            statistics[dim] = np.nan
            p_values[dim] = np.nan

    # 计算通过率（p > α表示不能拒绝H₀，即认为服从正态分布）
    valid_mask = np.isfinite(p_values)
    pass_mask = p_values > alpha
    pass_rate = pass_mask[valid_mask].sum() / valid_mask.sum() if valid_mask.any() else 0.0

    return {
        'statistics': statistics,
        'p_values': p_values,
        'pass_rate': pass_rate,
        'pass_mask': pass_mask,
        'alpha': alpha
    }


def compute_suspicion_scores_normal(residuals: np.ndarray, normal_params: dict) -> np.ndarray:
    """
    计算维度级别损伤指标（基于残差幅值，未标准化）

    Args:
        residuals: (N, D) 残差矩阵
        normal_params: dict with 'means'

    Returns:
        scores: (D,) 维度级别损伤指标向量，表示每维度的平均|残差|
    """
    N, D = residuals.shape
    means = normal_params['means']

    # 1. 去均值
    residuals_centered = residuals - means

    # 2. 计算每维度的平均|残差|
    scores = np.abs(residuals_centered).mean(axis=0)  # (D,)

    return scores


def compute_anomaly_scores(V: np.ndarray, model, device, batch_size: int = 512, method: str = 'mean') -> np.ndarray:
    """
    计算样本级异常分数

    Args:
        V: 输入数据 (N, D)
        model: 自编码器模型
        device: 计算设备
        batch_size: 批处理大小
        method: 'mean' = 全通道平均MAE（适合全局退化）
                'max'  = 最大通道MAE（适合局部损伤）

    Returns:
        scores: (N,) 每个样本的异常分数
    """
    residuals = compute_residuals(V, model, device, batch_size)
    abs_res = np.abs(residuals)
    if method == 'max':
        return abs_res.max(axis=1)
    else:
        return abs_res.mean(axis=1)


def compute_detection_metrics(
    scores_damage: np.ndarray,
    scores_control: np.ndarray,
    fpr_target: float = 0.05,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95
) -> dict:
    """
    计算损伤检测指标：ROC-AUC、TPR@FPR5%、最佳F1，以及AUC的Bootstrap置信区间

    Args:
        scores_damage: 损伤样本的异常分数 (N_damage,)
        scores_control: 正常样本的异常分数 (N_control,)
        fpr_target: 目标误报率（默认5%）
        n_bootstrap: Bootstrap重采样次数（默认1000）
        ci_level: 置信水平（默认0.95）

    Returns:
        dict: {'auc', 'auc_ci_lo', 'auc_ci_hi', 'tpr_at_fpr', 'best_f1', 'fprs', 'tprs'}
    """
    from sklearn.metrics import roc_auc_score, roc_curve, f1_score

    y_true = np.concatenate([np.ones(len(scores_damage)), np.zeros(len(scores_control))])
    y_score = np.concatenate([scores_damage, scores_control])

    auc = roc_auc_score(y_true, y_score)
    fprs, tprs, thresholds = roc_curve(y_true, y_score)

    idx = min(np.searchsorted(fprs, fpr_target), len(fprs) - 1)
    tpr_at_fpr = float(tprs[idx])

    best_f1 = max(
        f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
        for t in thresholds
    )

    # Bootstrap CI for AUC
    rng = np.random.RandomState(42)
    n_d, n_c = len(scores_damage), len(scores_control)
    auc_boots = []
    for _ in range(n_bootstrap):
        idx_d = rng.choice(n_d, n_d, replace=True)
        idx_c = rng.choice(n_c, n_c, replace=True)
        y_true_b = np.concatenate([np.ones(n_d), np.zeros(n_c)])
        y_score_b = np.concatenate([scores_damage[idx_d], scores_control[idx_c]])
        try:
            auc_boots.append(roc_auc_score(y_true_b, y_score_b))
        except ValueError:
            pass
    if len(auc_boots) >= 2:
        auc_lo = float(np.percentile(auc_boots, (1 - ci_level) / 2 * 100))
        auc_hi = float(np.percentile(auc_boots, (1 + ci_level) / 2 * 100))
    else:
        auc_lo = float(auc)
        auc_hi = float(auc)

    return {
        'auc': float(auc),
        'auc_ci_lo': auc_lo,
        'auc_ci_hi': auc_hi,
        'tpr_at_fpr': tpr_at_fpr,
        'best_f1': float(best_f1),
        'fprs': fprs,
        'tprs': tprs
    }


def smooth_scores(scores: np.ndarray, window: int) -> np.ndarray:
    """平滑损伤指标"""
    return uniform_filter1d(scores, size=window, mode="nearest")


# ========================================
# 可视化函数
# ========================================

def visualize_combined_residual_analysis(
    test_residuals: np.ndarray,
    output_dir: str,
    normal_params: dict,
    n_samples: int = 5,
    name: str = "combined_residual_analysis"
) -> None:
    """
    合并可视化：残差热力图 + 单样本折线图 + 平均残差曲线（未标准化）

    布局（3个Panel）：
    - Panel 1（上方）：残差幅度热力图 |residuals| (N_samples × D_dimensions)
    - Panel 2（中间）：单样本逐维度残差折线图（多样本叠加）
    - Panel 3（下方）：每个维度的平均|残差|曲线

    所有子图横轴对齐（250维度对齐）

    Args:
        test_residuals: (N, D) 测试数据残差矩阵
        output_dir: 输出目录
        normal_params: 健康基线参数字典
        n_samples: 要可视化的样本数（默认前5个）
        name: 文件名
    """
    N, D = test_residuals.shape
    n_samples = min(n_samples, N)
    abs_residuals = np.abs(test_residuals)
    mean_abs_residuals = abs_residuals.mean(axis=0)

    # 使用GridSpec实现紧凑布局：
    # - 5行给热力图
    # - 2行给折线图
    # - 2行给平均曲线
    fig = plt.figure(figsize=(24, 14))
    gs = GridSpec(9, 1, figure=fig, hspace=0.3)

    # ========== Panel 1: 残差幅度热力图（占5行）==========
    ax1 = fig.add_subplot(gs[0:5, 0])
    im = ax1.imshow(
        abs_residuals,
        aspect='auto',
        cmap='YlOrRd',
        interpolation='nearest',
        vmin=0,
        vmax=np.percentile(abs_residuals, 99.8)
    )
    ax1.set_ylabel('Sample Index', fontsize=26)

    # 添加网格线
    for i in range(50, D, 50):
        ax1.axvline(i - 0.5, color='white', linestyle='-', linewidth=0.8, alpha=0.7)

    # Adaptive y-tick interval to avoid overlap
    if N <= 100:
        ytick_step = 20
    elif N <= 300:
        ytick_step = 50
    elif N <= 600:
        ytick_step = 100
    else:
        ytick_step = 200

    for i in range(ytick_step, N, ytick_step):
        ax1.axhline(i - 0.5, color='white', linestyle='-', linewidth=0.8, alpha=0.7)

    ax1.set_yticks(np.arange(0, N, ytick_step))
    ax1.tick_params(labelsize=22)
    ax1.set_xlim([0, D - 1])

    # ========== Panel 2: 单样本折线图（占2行）==========
    ax2 = fig.add_subplot(gs[5:7, 0], sharex=ax1)

    sample_colors = [
        "#56DA3F",
        '#FFD93D',
        '#6C5CE7',
        '#00D2FF',
        "#A515BC"
    ]

    for sample_idx in range(n_samples):
        residuals_sample = np.abs(test_residuals[sample_idx, :])
        x_pos = np.arange(D)
        label = 'Different Samples' if sample_idx == 0 else None

        ax2.plot(
            x_pos,
            residuals_sample,
            color=sample_colors[sample_idx % len(sample_colors)],
            linewidth=0.6,
            linestyle='-',
            alpha=0.8,
            label=label
        )

    ax2.set_ylabel('|Residual|', fontsize=26)
    ax2.set_xlim([0, D - 1])
    ax2.tick_params(labelsize=22)

    # 添加垂直分隔线
    for i in range(50, D, 50):
        ax2.axvline(i, color='gray', linestyle=':', linewidth=0.5, alpha=0.4)

    # 添加图例（局部字号：整体 tick=22, 图例 14 保持梯度）
    ax2.legend(loc='upper right', frameon=False, fontsize=14)

    # ========== Panel 3: 平均残差曲线（占2行）==========
    ax3 = fig.add_subplot(gs[7:9, 0], sharex=ax1)
    ax3.plot(range(D), mean_abs_residuals, color='tab:blue', linewidth=1.0)
    ax3.fill_between(range(D), 0, mean_abs_residuals, color='tab:blue', alpha=0.2)
    ax3.set_xlabel('Dimension Index', fontsize=26)
    ax3.set_ylabel('Mean |Residual|', fontsize=26)
    ax3.tick_params(labelsize=22)

    # 添加垂直分隔线
    for i in range(50, D, 50):
        ax3.axvline(i, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

    # 设置共同的x轴刻度（确保250对齐）
    ax3.set_xticks(np.arange(0, D, 50))

    # 添加colorbar
    cbar = plt.colorbar(im, ax=[ax1, ax2, ax3], fraction=0.02, pad=0.01)
    cbar.set_label('|Residual|', fontsize=24)
    cbar.ax.tick_params(labelsize=20)

    # 调整colorbar位置，使其与热力图（ax1）高度对齐
    pos1 = ax1.get_position()  # 获取ax1的位置
    pos_cbar = cbar.ax.get_position()  # 获取colorbar的位置
    cbar.ax.set_position([pos_cbar.x0, pos1.y0, pos_cbar.width, pos1.height])

    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    img_path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(img_path, dpi=VIS_DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ========================================
# 3D渲染函数
# ========================================

def load_camera_position(config_path: str) -> list:
    """
    从JSON配置文件加载相机位置

    Args:
        config_path: camera_position.json文件路径

    Returns:
        相机位置列表 [camera_position, focal_point, view_up]
    """
    import json

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"\n{'='*60}\n"
            f"错误：相机位置配置文件不存在！\n"
            f"文件路径: {config_path}\n\n"
            f"请先运行 90_interactive_camera_setup.py 脚本设置相机视角：\n"
            f"  python script/90_interactive_camera_setup.py\n\n"
            f"该脚本将帮助您交互式地调整3D视角并保存到配置文件。\n"
            f"{'='*60}"
        )

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        camera_position = [
            config['camera_position'],
            config['focal_point'],
            config['view_up']
        ]

        return camera_position

    except KeyError as e:
        raise ValueError(
            f"相机配置文件格式错误，缺少必需字段: {e}\n"
            f"请重新运行 90_interactive_camera_setup.py 生成正确的配置文件"
        )
    except json.JSONDecodeError as e:
        raise ValueError(
            f"相机配置文件JSON格式错误: {e}\n"
            f"请检查文件内容或重新运行 90_interactive_camera_setup.py"
        )


def parse_elsets_from_inp(inp_file_path, target_elset=None):
    """从INP文件中提取元素集定义"""
    element_sets = {}
    current_elset_name = None
    current_elset_data = []
    is_generate = False

    with open(inp_file_path, 'r', encoding='latin1') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('**'):
                continue

            if line.upper().startswith('*ELSET'):
                if current_elset_name and current_elset_data:
                    element_sets[current_elset_name] = current_elset_data
                    if target_elset and current_elset_name == target_elset:
                        return element_sets

                current_elset_data = []
                is_generate = False
                match = re.search(r'elset\s*=\s*([^,\s]+)', line, re.IGNORECASE)
                if match:
                    current_elset_name = match.group(1).strip()
                    if 'generate' in line.lower():
                        is_generate = True
                else:
                    current_elset_name = None
                continue

            if line.startswith('*') and not line.startswith('**'):
                if current_elset_name and current_elset_data:
                    element_sets[current_elset_name] = current_elset_data
                    if target_elset and current_elset_name == target_elset:
                        return element_sets
                current_elset_name = None
                current_elset_data = []
                is_generate = False
                continue

            if current_elset_name:
                if is_generate:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 2:
                        try:
                            start = int(parts[0])
                            end = int(parts[1])
                            increment = int(parts[2]) if len(parts) >= 3 else 1
                            generated_ids = list(range(start, end + 1, increment))
                            current_elset_data.extend(generated_ids)
                        except ValueError:
                            pass
                else:
                    try:
                        ids = [int(x.strip()) for x in line.split(',') if x.strip()]
                        current_elset_data.extend(ids)
                    except ValueError:
                        pass

        if current_elset_name and current_elset_data:
            element_sets[current_elset_name] = current_elset_data

    return element_sets


def get_colormap_with_range(cmap_name: str, vmin: float = 0.0, vmax: float = 1.0):
    """创建范围限制的colormap"""
    if vmin == 0.0 and vmax == 1.0:
        return cmap_name
    try:
        original_cmap = plt.get_cmap(cmap_name)
        colors = original_cmap(np.linspace(vmin, vmax, 256))
        new_cmap = LinearSegmentedColormap.from_list(
            f'{cmap_name}_range_{vmin}_{vmax}',
            colors
        )
        return new_cmap
    except Exception:
        return cmap_name


def extract_middlewhole_submesh(base_mesh, middlewhole_ids, id_mapping):
    """从完整网格中提取middlewhole子网格"""
    total_cells = base_mesh.n_cells
    is_middlewhole = np.zeros(total_cells, dtype=bool)
    middlewhole_cell_indices = []

    for abaqus_id in middlewhole_ids:
        if abaqus_id in id_mapping:
            vtu_index = id_mapping[abaqus_id]
            if 0 <= vtu_index < total_cells:
                is_middlewhole[vtu_index] = True
                middlewhole_cell_indices.append(vtu_index)

    mesh_copy = base_mesh.copy()
    mesh_copy.cell_data['is_middlewhole'] = is_middlewhole.astype(float)
    middlewhole_mesh = mesh_copy.threshold(0.5, scalars='is_middlewhole')

    return middlewhole_mesh, middlewhole_cell_indices


def interpolate_to_nodes(base_mesh, middlewhole_mesh, damage_values, measure_ids, id_mapping):
    """将252个测点的损伤值插值到middlewhole区域的节点上"""
    from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator, RBFInterpolator

    # 获取测点的单元中心坐标
    cell_centers = base_mesh.cell_centers().points
    measure_centers = []
    measure_values = []

    for i, abaqus_id in enumerate(measure_ids):
        if abaqus_id in id_mapping:
            vtu_index = id_mapping[abaqus_id]
            if 0 <= vtu_index < len(cell_centers):
                measure_centers.append(cell_centers[vtu_index])
                measure_values.append(damage_values[i])

    measure_centers = np.array(measure_centers)
    measure_values = np.array(measure_values)

    # 获取middlewhole的节点坐标
    node_coords = middlewhole_mesh.points

    # 执行插值
    if INTERPOLATION_METHOD == "rbf":
        try:
            interpolator = RBFInterpolator(
                measure_centers,
                measure_values,
                kernel=RBF_KERNEL,
                smoothing=RBF_SMOOTHING,
                epsilon=RBF_EPSILON
            )
            node_values = interpolator(node_coords)
        except Exception:
            try:
                interpolator = LinearNDInterpolator(measure_centers, measure_values, fill_value=0.0)
                node_values = interpolator(node_coords)
            except Exception:
                interpolator = NearestNDInterpolator(measure_centers, measure_values)
                node_values = interpolator(node_coords)
    elif INTERPOLATION_METHOD == "linear":
        try:
            interpolator = LinearNDInterpolator(measure_centers, measure_values, fill_value=0.0)
            node_values = interpolator(node_coords)
        except Exception:
            interpolator = NearestNDInterpolator(measure_centers, measure_values)
            node_values = interpolator(node_coords)
    else:
        interpolator = NearestNDInterpolator(measure_centers, measure_values)
        node_values = interpolator(node_coords)

    return node_values


def render_damage_3d(base_mesh, middlewhole_mesh, node_values, output_path, camera_position,
                     override_clim_max=None):
    """
    渲染3D损伤检测图（使用未标准化的残差幅值）

    Args:
        base_mesh: 完整网格
        middlewhole_mesh: middlewhole子网格
        node_values: 节点损伤值
        output_path: 输出路径
        camera_position: 相机位置 [camera_position, focal_point, view_up]
        override_clim_max: 可选，强制统一色标上限（用于跨配置对比）
    """
    if not HAS_PYVISTA:
        print("  [跳过] PyVista未安装，跳过3D渲染")
        return

    plotter = pv.Plotter(
        window_size=(RENDER_WINDOW_WIDTH, RENDER_WINDOW_HEIGHT),
        off_screen=True
    )
    plotter.set_background(RENDER_BACKGROUND)

    # 添加middlewhole区域
    mesh_with_data = middlewhole_mesh.copy()
    mesh_with_data.point_data['damage'] = node_values

    data_min = node_values.min()
    data_max = node_values.max()
    clim_min = 0.0
    clim_max = override_clim_max if override_clim_max is not None else max(data_max, 0.5)

    colormap = get_colormap_with_range(COLORMAP_NAME, COLORMAP_RANGE[0], COLORMAP_RANGE[1])
    colorbar_title = 'Mean |Residual|'

    # 3D渲染字号：每个子图 1920px 宽，6张拼成 ~5890px，论文中缩放约 0.31x
    # 需要 ~60pt 标题和 ~52pt 标签才能在论文中显示为 ~18pt / ~16pt
    sbar_title_fontsize = 60
    sbar_label_fontsize = 52

    if USE_DISCRETE_COLORMAP:
        plotter.add_mesh(
            mesh_with_data,
            scalars='damage',
            cmap=colormap,
            n_colors=DISCRETE_COLORMAP_BANDS,
            clim=(clim_min, clim_max),
            show_edges=False,
            opacity=1.0,
            show_scalar_bar=True,
            scalar_bar_args={
                'title': colorbar_title,
                'vertical': True,
                'position_x': 0.82,
                'position_y': 0.15,
                'width': 0.05,
                'height': 0.7,
                'n_labels': min(DISCRETE_COLORMAP_BANDS + 1, 11),
                'title_font_size': sbar_title_fontsize,
                'label_font_size': sbar_label_fontsize,
                'font_family': 'times'
            }
        )
    else:
        plotter.add_mesh(
            mesh_with_data,
            scalars='damage',
            cmap=colormap,
            clim=(clim_min, clim_max),
            show_edges=False,
            opacity=1.0,
            show_scalar_bar=True,
            scalar_bar_args={
                'title': colorbar_title,
                'vertical': True,
                'position_x': 0.82,
                'position_y': 0.15,
                'width': 0.05,
                'height': 0.7,
                'title_font_size': sbar_title_fontsize,
                'label_font_size': sbar_label_fontsize,
                'font_family': 'times'
            }
        )

    # 添加特征边线
    feature_edges = base_mesh.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=True,
        feature_edges=True,
        manifold_edges=False,
        feature_angle=FEATURE_EDGE_ANGLE
    )
    plotter.add_mesh(
        feature_edges,
        color=FEATURE_EDGE_COLOR,
        line_width=FEATURE_EDGE_WIDTH,
        render_lines_as_tubes=False
    )

    # 设置相机位置（从配置文件加载）
    plotter.camera_position = camera_position

    # 保存图片
    plotter.screenshot(output_path)
    plotter.close()


# ========================================
# 图像合并函数
# ========================================

def merge_images_horizontal(left_path: str, right_path: str, output_path: str):
    """
    水平合并两张图片（左右拼接，无间距）

    Args:
        left_path: 左侧图片路径
        right_path: 右侧图片路径
        output_path: 输出路径
    """
    if not HAS_PIL:
        print("  [警告] PIL未安装，跳过图像合并")
        return

    if not os.path.exists(left_path) or not os.path.exists(right_path):
        print(f"  [警告] 图片不存在，跳过合并")
        return

    # 读取两张图片
    img_left = Image.open(left_path)
    img_right = Image.open(right_path)

    # 获取尺寸
    width_left, height_left = img_left.size
    width_right, height_right = img_right.size

    # 使用最大高度，保持原始宽度
    max_height = max(height_left, height_right)

    # 创建新画布（左右拼接，无间距）
    merged_width = width_left + width_right
    merged_img = Image.new('RGB', (merged_width, max_height), 'white')

    # 粘贴图片（无间距）
    merged_img.paste(img_left, (0, 0))
    merged_img.paste(img_right, (width_left, 0))

    # 保存
    merged_img.save(output_path, dpi=(VIS_DPI, VIS_DPI))


def merge_images_grid(image_paths: List[List[str]], output_path: str,
                      column_titles: Optional[List[str]] = None,
                      row_titles: Optional[List[str]] = None,
                      border_width: int = 0):
    """
    合并图片为网格布局（如2x3，2xn等）

    Args:
        image_paths: 2D列表，[[row1_img1, row1_img2, ...], [row2_img1, row2_img2, ...], ...]
        output_path: 输出路径
        column_titles: 可选，列标题列表（长度应与列数相同）
        row_titles: 可选，行标题列表（长度应与行数相同），在左侧垂直渲染
    """
    if not HAS_PIL:
        print("  [警告] PIL未安装，跳过图像合并")
        return

    # 检查所有图片是否存在
    for row in image_paths:
        for img_path in row:
            if not os.path.exists(img_path):
                print(f"  [警告] 图片不存在: {img_path}，跳过合并")
                return

    # 加载所有图片
    images = []
    for row in image_paths:
        row_images = [Image.open(img_path) for img_path in row]
        images.append(row_images)

    # 计算每行的最大高度和每列的最大宽度
    n_rows = len(images)
    n_cols = len(images[0])

    row_heights = []
    for row in images:
        max_height = max(img.size[1] for img in row)
        row_heights.append(max_height)

    col_widths = []
    for col_idx in range(n_cols):
        max_width = max(images[row_idx][col_idx].size[0] for row_idx in range(n_rows))
        col_widths.append(max_width)

    # 计算标题区域高度（字号增大，预留空间也需增大）
    title_height = 0
    if column_titles and len(column_titles) == n_cols:
        title_height = 200  # 为标题预留的高度

    # 计算行标题区域宽度
    row_title_width = 0
    if row_titles and len(row_titles) == n_rows:
        row_title_width = 260  # 为行标题预留的宽度（增大以确保旋转文字可读）

    # 准备字体（列标题和行标题共用）
    # 字号增大: 3D图 ~5890px 宽 → 论文中 0.31x → 需要 ~100px 字号 → 论文中 ~31px ≈ ~10pt
    font = None
    font_size = 100
    if (column_titles and len(column_titles) == n_cols) or (row_titles and len(row_titles) == n_rows):
        # 尝试多种Times New Roman字体路径
        times_fonts = [
            "times.ttf",           # Windows
            "Times New Roman.ttf", # Windows
            "timesnewroman.ttf",
            "C:/Windows/Fonts/times.ttf",
            "C:/Windows/Fonts/Times New Roman.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",  # Linux
        ]
        for font_path in times_fonts:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue
        # 如果Times New Roman不可用，回退到其他字体
        if font is None:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                except:
                    font = ImageFont.load_default()

    # 创建新画布
    total_width = sum(col_widths) + row_title_width
    total_height = sum(row_heights) + title_height
    merged_img = Image.new('RGB', (total_width, total_height), 'white')

    # 绘制列标题
    if column_titles and len(column_titles) == n_cols:
        draw = ImageDraw.Draw(merged_img)

        x_offset = row_title_width
        for col_idx, title in enumerate(column_titles):
            # 计算文字位置（居中）
            bbox = draw.textbbox((0, 0), title, font=font)
            text_width = bbox[2] - bbox[0]
            text_x = x_offset + (col_widths[col_idx] - text_width) // 2
            text_y = (title_height - (bbox[3] - bbox[1])) // 2

            # 绘制文字
            draw.text((text_x, text_y), title, fill='black', font=font)
            x_offset += col_widths[col_idx]

    # 绘制行标题（垂直文字，在图片左侧）
    if row_titles and len(row_titles) == n_rows:
        y_offset = title_height
        for row_idx, rtitle in enumerate(row_titles):
            # 创建临时图像用于旋转文字
            tmp_draw = ImageDraw.Draw(merged_img)
            bbox = tmp_draw.textbbox((0, 0), rtitle, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            # 在透明临时图上绘制文字，然后旋转90度
            txt_img = Image.new('RGBA', (text_w + 20, text_h + 20), (255, 255, 255, 0))
            txt_draw = ImageDraw.Draw(txt_img)
            txt_draw.text((10, 10), rtitle, fill='black', font=font)
            txt_rotated = txt_img.rotate(90, expand=True)

            # 居中粘贴到行标题区域
            paste_x = (row_title_width - txt_rotated.width) // 2
            paste_y = y_offset + (row_heights[row_idx] - txt_rotated.height) // 2
            merged_img.paste(txt_rotated, (paste_x, paste_y), txt_rotated)

            y_offset += row_heights[row_idx]

    # 粘贴图片
    y_offset = title_height
    for row_idx, row in enumerate(images):
        x_offset = row_title_width
        for col_idx, img in enumerate(row):
            merged_img.paste(img, (x_offset, y_offset))
            if border_width > 0:
                draw_border = ImageDraw.Draw(merged_img)
                draw_border.rectangle(
                    [x_offset, y_offset,
                     x_offset + col_widths[col_idx] - 1,
                     y_offset + row_heights[row_idx] - 1],
                    outline="black", width=border_width,
                )
            x_offset += col_widths[col_idx]
        y_offset += row_heights[row_idx]

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_img.save(output_path, dpi=(VIS_DPI, VIS_DPI))
