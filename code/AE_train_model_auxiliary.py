"""
AE_train_model_auxiliary.py
============================
Core AE training module: Autoencoder class definition, train_model(),
freeze-strategy training, plotting utilities, and data loading.
Imported by all AE stage scripts; not executed directly.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neighbors import KernelDensity
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset



# ========================================
# 常量
# ========================================

FIG_DPI = 300  # 图片输出DPI

# 绘图风格（学术风）
# 字号梯度（硬约束）：轴标题/标签 (20) > 刻度 (16) > 图例 (12)
PLOT_STYLE = {
    "font.family": "Times New Roman",
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 12,
    "legend.frameon": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.grid": False,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.bottom": True,
    "axes.spines.left": True,
}


# ========================================
# 绘图辅助函数
# ========================================

def apply_style(plot_style: dict | None = None):
    """应用学术风绘图样式"""
    if plot_style is None:
        plot_style = PLOT_STYLE
    plt.rcParams.update(plot_style)


def safe_kde(data: np.ndarray, grid: np.ndarray | None = None) -> tuple[np.ndarray | None, np.ndarray | None]:
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


# ========================================
# 模型定义
# ========================================

class Autoencoder(nn.Module):
    """最基础的多层感知机自编码器"""

    def __init__(
        self,
        input_dim: int,
        encoder_dims: list[int],
        latent_dim: int,
        decoder_dims: list[int],
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim

        act_cls = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }.get(activation, nn.ReLU)

        # 编码器
        enc: list[nn.Module] = []
        prev = input_dim
        for h in encoder_dims:
            enc += [nn.Linear(prev, h), act_cls()]
            if dropout and dropout > 0:
                enc += [nn.Dropout(dropout)]
            prev = h
        enc += [nn.Linear(prev, latent_dim)]
        self.encoder = nn.Sequential(*enc)

        # 解码器
        dec: list[nn.Module] = []
        prev = latent_dim
        for h in decoder_dims:
            dec += [nn.Linear(prev, h), act_cls()]
            if dropout and dropout > 0:
                dec += [nn.Dropout(dropout)]
            prev = h
        dec += [nn.Linear(prev, input_dim)]
        self.decoder = nn.Sequential(*dec)

        # Xavier 初始化线性层
        def _init(m: nn.Module):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def reinit_decoder_weights(model: Autoencoder) -> None:
    """将解码器所有 Linear 层重初始化为 Xavier 随机，编码器不变。"""
    def _init(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    model.decoder.apply(_init)


# ========================================
# 冻结策略实现
# ========================================

def apply_freeze_strategy(model: Autoencoder, strategy: str, verbose: bool = True):
    """
    应用冻结策略

    Args:
        model: 自编码器模型
        strategy: 冻结策略名称
            - 'full' or 'none': 全量微调（不冻结任何层）
            - 'freeze_encoder' or 'encoder': 冻结编码器（仅训练解码器）
            - 'freeze_bottom' or 'bottom': 冻结底层（冻结encoder前2个线性层）
        verbose: 是否打印冻结信息
    """
    # 先解冻所有参数
    for param in model.parameters():
        param.requires_grad = True

    # 处理策略名称的不同变体
    strategy_normalized = strategy.lower().replace('freeze_', '').replace('_', '')

    if strategy_normalized in ['full', 'none']:
        # 策略1：全量微调
        if verbose:
            print("[冻结策略] Full Fine-tuning - 训练所有层")

    elif strategy_normalized == 'encoder':
        # 策略2：冻结编码器
        for param in model.encoder.parameters():
            param.requires_grad = False
        if verbose:
            print("[冻结策略] Freeze Encoder - 冻结编码器，仅训练解码器")

    elif strategy_normalized == 'bottom':
        # 策略3：冻结底层
        encoder_layers = list(model.encoder.children())
        freeze_indices = [0, 3]  # 第1个和第2个Linear层

        for idx in freeze_indices:
            if idx < len(encoder_layers):
                for param in encoder_layers[idx].parameters():
                    param.requires_grad = False

        if verbose:
            print("[冻结策略] Freeze Bottom Layers - 冻结encoder前2个线性层，训练encoder后续层+decoder")

    elif strategy_normalized in ['lastlayer', 'last_layer']:
        # 策略4：仅训练解码器最后一个 Linear 层
        for param in model.parameters():
            param.requires_grad = False
        # 找到解码器最后一个 Linear 层并解冻
        decoder_layers = list(model.decoder.children())
        for layer in reversed(decoder_layers):
            if isinstance(layer, nn.Linear):
                for param in layer.parameters():
                    param.requires_grad = True
                break
        if verbose:
            print("[冻结策略] Last Layer Only - 仅训练decoder最后一个Linear层")

    elif strategy in ['two_stage_1', 'two_stage_2']:
        # 两阶段训练（向后兼容，已废弃）
        if strategy == 'two_stage_1':
            for param in model.encoder.parameters():
                param.requires_grad = False
            if verbose:
                print("[冻结策略] Two-Stage Stage-1 - 冻结encoder，训练decoder")
        else:
            if verbose:
                print("[冻结策略] Two-Stage Stage-2 - 解冻所有层，全量微调")

    else:
        raise ValueError(f"Unknown freeze strategy: {strategy}")

    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"[参数统计] 可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")


# ========================================
# 数据加载
# ========================================

def load_data_from_path(data_dir: str, val_samples: int, data_name: str = "数据") -> Tuple[np.ndarray, np.ndarray]:
    """
    从指定路径加载数据，固定验证集，其余为训练集

    Args:
        data_dir: 数据目录路径（04脚本输出）
        val_samples: 验证集样本数
        data_name: 数据名称（用于日志）

    Returns:
        (V_train, V_val): 训练集和验证集
    """
    npz_path = os.path.join(data_dir, 'preprocessed_data_raw.npz')
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"未找到{data_name}: {npz_path}")

    data = np.load(npz_path)
    V = data["V"].astype(np.float32)
    print(f"[加载] {data_name}形状: {V.shape}")

    # 固定验证集，其余为训练集
    N = V.shape[0]
    if N <= val_samples:
        raise ValueError(f"{data_name}数据量({N})不足以划分{val_samples}个验证集样本")

    V_train = V[:N - val_samples]
    V_val = V[N - val_samples:]

    print(f"[划分] 训练集: {V_train.shape[0]}, 验证集: {V_val.shape[0]}")
    return V_train, V_val


# ========================================
# 训练核心
# ========================================

def _warmup_cosine_lambda(epoch: int, warmup: int, total: int, eta_min_ratio: float) -> float:
    """Linear warmup for `warmup` epochs, then cosine decay to eta_min_ratio.

    LR schedule:
        epoch < warmup:  lr = base_lr * (epoch+1) / warmup   (linear ramp)
        epoch >= warmup: lr = base_lr * [eta_min_ratio + 0.5*(1-eta_min_ratio)*(1+cos(pi*progress))]
    """
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup)
    return eta_min_ratio + 0.5 * (1.0 - eta_min_ratio) * (1.0 + math.cos(math.pi * progress))


def train_model(
    V_train: np.ndarray,
    V_val: np.ndarray,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    output_dir: str,
    ae_config: dict,  # {'encoder_dims', 'latent_dim', 'decoder_dims', 'dropout', 'activation'}
    training_config: dict,  # {'weight_decay', 'grad_clip', 'use_lr_scheduler', 'lr_scheduler_type', etc.}
    plot_style: dict,
    fig_dpi: int,
    pretrain_model_path: Optional[str] = None,
    model_name: str = "model",
    max_train_samples: Optional[int] = None,
    train_shuffle: bool = True,
    l2_sp_alpha: float = 0.0,
    noise_std: float = 0.0,
    reinit_decoder: bool = False,
    encoder_lr: Optional[float] = None,
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    统一训练函数

    Args:
        V_train: 训练数据 (N, D)
        V_val: 验证数据 (M, D)
        device: 训练设备
        epochs: 训练轮数
        lr: 学习率
        batch_size: 批大小
        output_dir: 输出目录
        ae_config: 自编码器配置字典
        training_config: 训练配置字典
        plot_style: 绘图样式字典
        fig_dpi: 图像DPI
        pretrain_model_path: 预训练模型路径（None=从头训练）
        model_name: 模型名称
        max_train_samples: 最大训练样本数
        train_shuffle: 是否打乱训练数据
        l2_sp_alpha: L2-SP 正则化强度 (>=0)。向预训练权重施加 L2 惩罚，
                     防止微调时参数偏离太远（灾难性遗忘）。0=不使用。
        noise_std: 训练时输入高斯噪声标准差 (>=0)。去噪 AE 风格增强，
                   在输入上加噪声，重构目标为干净输入。0=不使用。

    Returns:
        (model, train_losses, val_losses)
    """
    print(f"\n{'='*60}")
    print(f"训练 {model_name}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    # 限制训练数据量（用于消融实验）
    if max_train_samples is not None and max_train_samples < V_train.shape[0]:
        V_train = V_train[:max_train_samples]
        print(f"[数据量消融] 使用前{max_train_samples}个训练样本")

    D = V_train.shape[1]
    X_train = torch.from_numpy(V_train).to(device)
    X_val = torch.from_numpy(V_val).to(device)

    # 初始化模型
    model = Autoencoder(
        D,
        ae_config['encoder_dims'],
        ae_config['latent_dim'],
        ae_config['decoder_dims'],
        ae_config['dropout'],
        ae_config['activation']
    ).to(device)

    # 加载预训练权重（迁移学习）
    if pretrain_model_path:
        print(f"[迁移学习] 加载预训练权重: {pretrain_model_path}")
        checkpoint = torch.load(pretrain_model_path, map_location=device)
        model.load_state_dict(checkpoint)

    # 解码器重初始化（保留编码器预训练权重，解码器 Xavier 随机）
    if reinit_decoder and pretrain_model_path:
        reinit_decoder_weights(model)
        print(f"[解码器重初始化] 解码器已重置为 Xavier 随机，编码器保持预训练")

    # L2-SP: 存储预训练权重副本（在冻结策略之前）
    _l2sp_ref: Dict[str, torch.Tensor] = {}
    if l2_sp_alpha > 0 and pretrain_model_path:
        for n, p in model.named_parameters():
            _l2sp_ref[n] = p.clone().detach()
        print(f"[L2-SP] alpha={l2_sp_alpha:.4f}, 参考权重已缓存")

    # 解码器重初始化时也自动回退到 Adam（解码器从零训练）
    optimizer_type = training_config.get('optimizer_type', 'adam').lower()
    if not pretrain_model_path and optimizer_type != 'adam':
        print(f"[优化器] 从头训练 -> 自动回退到 Adam (配置的 {optimizer_type} 仅用于微调)")
        optimizer_type = 'adam'
    # 注：reinit_decoder 不再强制 Adam，由用户配置决定
    weight_decay = training_config['weight_decay']

    # 差异化学习率：encoder_lr != None 时为编码器和解码器设不同 LR
    if encoder_lr is not None and encoder_lr != lr:
        if encoder_lr == 0:
            # encoder_lr=0 等同于冻结编码器
            for p in model.encoder.parameters():
                p.requires_grad = False
            param_groups = [{"params": list(model.decoder.parameters()), "lr": lr}]
        else:
            param_groups = [
                {"params": list(model.encoder.parameters()), "lr": encoder_lr},
                {"params": list(model.decoder.parameters()), "lr": lr},
            ]
        if optimizer_type == 'sgd':
            momentum = training_config.get('sgd_momentum', 0.9)
            optimizer = SGD(param_groups, lr=lr, weight_decay=weight_decay, momentum=momentum)
            print(f"[优化器] SGD 差异LR (encoder={encoder_lr:.1e}, decoder={lr:.1e})")
        elif optimizer_type == 'adamw':
            optimizer = AdamW(param_groups, lr=lr, weight_decay=weight_decay)
            print(f"[优化器] AdamW 差异LR (encoder={encoder_lr:.1e}, decoder={lr:.1e})")
        else:
            optimizer = Adam(param_groups, lr=lr, weight_decay=weight_decay)
            print(f"[优化器] Adam 差异LR (encoder={encoder_lr:.1e}, decoder={lr:.1e})")
    else:
        if optimizer_type == 'sgd':
            momentum = training_config.get('sgd_momentum', 0.9)
            optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
            print(f"[优化器] SGD (momentum={momentum})")
        elif optimizer_type == 'adamw':
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            print(f"[优化器] AdamW")
        else:
            optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            print(f"[优化器] Adam")

    criterion = nn.MSELoss()

    # 初始化学习率调度器
    # warmup 仅在加载预训练权重时生效（从头训练无 momentum 失配问题）
    warmup_epochs = training_config.get('warmup_epochs', 0)
    if not pretrain_model_path:
        warmup_epochs = 0

    scheduler = None
    if training_config['use_lr_scheduler']:
        if training_config['lr_scheduler_type'] == 'cosine':
            eta_min = training_config.get('cosine_eta_min', 1e-6)
            if warmup_epochs > 0:
                eta_min_ratio = eta_min / lr
                scheduler = LambdaLR(
                    optimizer,
                    lr_lambda=lambda ep, w=warmup_epochs, t=epochs, r=eta_min_ratio: _warmup_cosine_lambda(ep, w, t, r)
                )
                print(f"[调度器] Warmup({warmup_epochs}) + Cosine -> eta_min={eta_min}")
            else:
                t_max = training_config.get('cosine_t_max') or epochs
                scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
                print(f"[调度器] CosineAnnealing: T_max={t_max}, eta_min={eta_min}")
        elif training_config['lr_scheduler_type'] == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min',
                factor=training_config.get('plateau_factor', 0.5),
                patience=training_config.get('plateau_patience', 50),
                verbose=True
            )
            print(f"[调度器] ReduceLROnPlateau")

    # 噪声增强：从 training_config 读取（优先使用函数参数）
    if noise_std == 0:
        noise_std = training_config.get('noise_std', 0.0)
    if noise_std > 0:
        print(f"[噪声增强] std={noise_std:.4f}")

    train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=train_shuffle)
    val_loader = DataLoader(TensorDataset(X_val), batch_size=batch_size, shuffle=False)

    print(f"开始训练 (共 {epochs} 轮)...")
    print("-" * 60)

    best_val = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # 训练
        model.train()
        running = 0.0
        count = 0
        for (xb,) in train_loader:
            optimizer.zero_grad(set_to_none=True)
            # 噪声增强：输入加高斯噪声，重构目标为干净输入
            if noise_std > 0:
                xb_input = xb + torch.randn_like(xb) * noise_std
            else:
                xb_input = xb
            recon = model(xb_input)
            loss = criterion(recon, xb)
            # L2-SP: 惩罚参数偏离预训练权重
            if _l2sp_ref:
                l2sp_term = sum(
                    (p - _l2sp_ref[n]).pow(2).sum()
                    for n, p in model.named_parameters()
                    if p.requires_grad and n in _l2sp_ref
                )
                loss = loss + l2_sp_alpha * l2sp_term
            loss.backward()
            if training_config['grad_clip'] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), training_config['grad_clip'])
            optimizer.step()
            running += loss.item() * xb.size(0)
            count += xb.size(0)
        train_epoch = running / max(1, count)

        # 验证
        model.eval()
        vrunning = 0.0
        vcount = 0
        with torch.no_grad():
            for (xb,) in val_loader:
                recon = model(xb)
                vloss = criterion(recon, xb)
                vrunning += vloss.item() * xb.size(0)
                vcount += xb.size(0)
        val_epoch = vrunning / max(1, vcount)

        train_losses.append(train_epoch)
        val_losses.append(val_epoch)

        # 学习率调度
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_epoch)
            else:
                scheduler.step()

        # 每轮打印
        if training_config['use_lr_scheduler']:
            print(f"Epoch {epoch+1:4d}/{epochs} | Train Loss: {train_epoch:.6f} | Val Loss: {val_epoch:.6f} | LR: {current_lr:.2e}")
        else:
            print(f"Epoch {epoch+1:4d}/{epochs} | Train Loss: {train_epoch:.6f} | Val Loss: {val_epoch:.6f}")

        # 保存最佳模型
        if val_epoch + 1e-9 < best_val:
            best_val = val_epoch
            torch.save(model.state_dict(), os.path.join(output_dir, "autoencoder.pth"))

    print("-" * 60)
    print(f"训练完成！最佳验证损失: {best_val:.6f}\n")

    # 保存损失日志
    pd.DataFrame({"train_loss": train_losses, "val_loss": val_losses}).to_csv(
        os.path.join(output_dir, "training_losses.csv"), index=False
    )

    # 绘制训练曲线
    plot_training_curve(train_losses, val_losses, output_dir, plot_style, fig_dpi)

    # 重新加载最佳模型
    model.load_state_dict(torch.load(os.path.join(output_dir, "autoencoder.pth"), map_location=device))

    return model, train_losses, val_losses


#  ========================================
# 绘图函数
# ========================================

def plot_training_curve(
    train_losses: List[float],
    val_losses: List[float],
    output_dir: str,
    plot_style: dict,
    fig_dpi: int,
    is_two_stage: bool = False,
    two_stage_epoch_1: int = 0
):
    """绘制训练曲线（双轴：正常坐标+对数坐标）"""
    apply_style(plot_style)
    xs = list(range(1, len(train_losses) + 1))

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)
    ax2 = ax1.twinx()

    # 线宽统一减细
    lw = 1.2

    # 左轴：正常坐标（实线）
    line1 = ax1.plot(xs, train_losses, label="Train", color="tab:blue", linewidth=lw, linestyle='-')
    line2 = ax1.plot(xs, val_losses, label="Validation", color="tab:orange", linewidth=lw, linestyle='-')

    # 右轴：对数坐标（虚线）
    ax2.plot(xs, train_losses, color="tab:blue", linewidth=lw * 0.8, linestyle='--', alpha=0.5)
    ax2.plot(xs, val_losses, color="tab:orange", linewidth=lw * 0.8, linestyle='--', alpha=0.5)

    if is_two_stage:
        ax1.axvline(two_stage_epoch_1, color='red', linestyle=':', linewidth=1.0, alpha=0.7)
        ax2.axvline(two_stage_epoch_1, color='red', linestyle=':', linewidth=1.0, alpha=0.7)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MSE, Linear Scale)", color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    ax2.set_ylabel("Loss (MSE, Log Scale)", color='black')
    ax2.set_yscale("log")
    ax2.tick_params(axis='y', labelcolor='black')

    from matplotlib.lines import Line2D
    legend_elements = line1 + line2 + [
        Line2D([0], [0], color='gray', linewidth=lw, linestyle='-', label='Linear Scale (Left Axis)'),
        Line2D([0], [0], color='gray', linewidth=lw * 0.8, linestyle='--', alpha=0.5, label='Log Scale (Right Axis)')
    ]
    legend_labels = ['Train', 'Validation', 'Linear Scale (Left Axis)', 'Log Scale (Right Axis)']

    if is_two_stage:
        legend_elements.append(Line2D([0], [0], color='red', linestyle=':', linewidth=1.0, alpha=0.7))
        legend_labels.append(f'Stage Boundary (Epoch {two_stage_epoch_1})')

    fig.legend(legend_elements, legend_labels,
               loc="outside upper center",
               ncol=min(len(legend_labels), 5),
               frameon=False, fontsize=12)

    fig.savefig(os.path.join(output_dir, "training_curve.png"), dpi=fig_dpi)
    plt.close(fig)


def plot_dual_comparison_curves(
    old_val_losses: List[float],
    transfer_val_losses: List[float],
    scratch_val_losses: List[float],
    output_dir: str,
    plot_style: dict,
    fig_dpi: int,
    filename: str = "training_comparison_curve.png",
    old_train_losses: Optional[List[float]] = None,
    transfer_train_losses: Optional[List[float]] = None,
    scratch_train_losses: Optional[List[float]] = None
):
    """
    绘制完整对比曲线（三模型：Pre-train, Transfer Learning, From Scratch）
    单轴 log-scale，Morandi 配色，无背景色块。
    """
    # Morandi palette (consistent with bar chart / ROC)
    C_PRETRAIN = "#C97A6C"   # dusty rose
    C_TL       = "#7BA7BC"   # blue-gray
    C_FS       = "#AAAAAA"   # gray

    # 整体字号放大：轴标签 24, 刻度 20, 图例 14（保持图例 < 刻度的梯度）
    curve_style = plot_style.copy()
    curve_style.update({
        "font.size": 24,
        "axes.labelsize": 24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 14,
    })
    apply_style(curve_style)
    fig, ax = plt.subplots(1, 1, figsize=(18, 5.6), constrained_layout=True)

    epochs_old = list(range(1, len(old_val_losses) + 1))
    epochs_transfer = list(range(len(old_val_losses) + 1,
                                 len(old_val_losses) + len(transfer_val_losses) + 1))
    epochs_scratch = list(range(len(old_val_losses) + 1,
                                len(old_val_losses) + len(scratch_val_losses) + 1))

    new_train_start = len(old_val_losses) + 0.5
    new_train_end = len(old_val_losses) + max(len(transfer_val_losses),
                                               len(scratch_val_losses))
    ax.set_xlim(1, new_train_end)

    lw = 1.2

    # Phase boundary: vertical dashed line
    ax.axvline(new_train_start, color="#BBBBBB", linewidth=1.0, linestyle="--")

    # Validation loss curves (solid)
    ax.plot(epochs_old, old_val_losses, label="Pre-train",
            color=C_PRETRAIN, linewidth=lw, linestyle="-")
    ax.plot(epochs_transfer, transfer_val_losses, label="Transfer Learning",
            color=C_TL, linewidth=lw, linestyle="-")
    ax.plot(epochs_scratch, scratch_val_losses, label="From Scratch",
            color=C_FS, linewidth=lw, linestyle="-")

    # Train loss curves (dashed, lighter)
    if old_train_losses:
        ax.plot(epochs_old, old_train_losses,
                color=C_PRETRAIN, linewidth=lw * 0.7, linestyle="--", alpha=0.45)
    if transfer_train_losses:
        ax.plot(epochs_transfer, transfer_train_losses,
                color=C_TL, linewidth=lw * 0.7, linestyle="--", alpha=0.45)
    if scratch_train_losses:
        ax.plot(epochs_scratch, scratch_train_losses,
                color=C_FS, linewidth=lw * 0.7, linestyle="--", alpha=0.45)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_yscale("log")

    # Phase labels (use axes fraction for y, data coords for x)
    pre_center_x = len(old_val_losses) * 0.5
    ft_center_x = new_train_start + (new_train_end - new_train_start) * 0.5
    import matplotlib.transforms as mtransforms
    blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(pre_center_x, 0.92, "Pre-train", ha="center", fontsize=22,
            color="black", transform=blend)
    ax.text(ft_center_x, 0.92, "Fine-tuning", ha="center", fontsize=22,
            color="black", transform=blend)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=C_PRETRAIN, lw=lw, label="Pre-train"),
        Line2D([0], [0], color=C_TL, lw=lw, label="Transfer Learning"),
        Line2D([0], [0], color=C_FS, lw=lw, label="From Scratch"),
        Line2D([0], [0], color="black", lw=lw, linestyle="-", label="Val loss"),
        Line2D([0], [0], color="black", lw=lw * 0.7, linestyle="--",
               alpha=0.45, label="Train loss"),
    ]
    fig.legend(handles=legend_elements,
               loc="outside upper center",
               ncol=5, frameon=False, fontsize=14)

    fig.savefig(os.path.join(output_dir, filename), dpi=fig_dpi)
    plt.close(fig)

    apply_style(plot_style)
    print(f"[保存] {filename}")

# ========================================
# 消融实验对比曲线（统一绘图函数）
# ========================================

# 消融实验预定义颜色方案
ABLATION_COLORS = {
    # 冻结策略消融
    'Full Fine-tuning': 'tab:blue',
    'Freeze Bottom': 'tab:orange',
    'Freeze Encoder': 'tab:green',
    # 数据量消融
    50: 'tab:blue',
    100: 'tab:orange',
    200: 'tab:green',
    400: 'tab:red',
}


def plot_ablation_comparison_curve(
    ablation_results: Dict,
    output_dir: str,
    ablation_type: str,
    custom_colors: Optional[Dict] = None,
    legend_loc: str = 'upper right',
    legend_ncol: int = 5
):
    """
    统一的消融实验对比曲线绘图函数（双轴叠加风格）

    使用左右双轴同时显示正常坐标和对数坐标

    Args:
        ablation_results: {label: (train_losses, val_losses)}
            - 冻结策略消融: {strategy_name: (train_losses, val_losses)}
            - 数据量消融: {n_samples: (train_losses, val_losses)}
        output_dir: 输出目录
        ablation_type: 消融类型
            - "freeze": 冻结策略消融 → 输出 ablation_freeze_comparison_curve.png
            - "datasize": 数据量消融 → 输出 ablation_datasize_comparison_curve.png
        custom_colors: 自定义颜色映射（可选）
        legend_loc: 图例位置
        legend_ncol: 图例列数
    """
    apply_style()
    fig, ax1 = plt.subplots(1, 1, figsize=(20, 6.6), constrained_layout=True)

    # 创建第二个y轴（共享x轴）
    ax2 = ax1.twinx()

    # 线宽统一减细
    lw = 1.2

    # 使用自定义颜色或默认颜色
    colors = custom_colors if custom_colors else ABLATION_COLORS

    lines = []

    # 确定迭代顺序（数值类型则排序，字符串类型保持原序）
    keys = ablation_results.keys()
    if all(isinstance(k, (int, float)) for k in keys):
        sorted_keys = sorted(keys)
    else:
        sorted_keys = list(keys)

    # 在左轴(ax1)绘制正常坐标曲线（实线）
    for key in sorted_keys:
        train_losses, val_losses = ablation_results[key]
        xs = list(range(1, len(val_losses) + 1))
        color = colors.get(key, 'gray')
        # 数值类型key使用 n=xxx 格式，字符串类型直接使用
        label = f'n={key}' if isinstance(key, (int, float)) else key
        line = ax1.plot(xs, val_losses, label=label, color=color, linewidth=lw, linestyle='-')
        lines.extend(line)

    # 在右轴(ax2)绘制对数坐标曲线（虚线，颜色相同但略浅）
    for key in sorted_keys:
        train_losses, val_losses = ablation_results[key]
        xs = list(range(1, len(val_losses) + 1))
        color = colors.get(key, 'gray')
        ax2.plot(xs, val_losses, color=color, linewidth=lw * 0.8, linestyle='--', alpha=0.5)

    # 设置左轴（正常坐标）
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss (MSE, Linear Scale)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # 设置右轴（对数坐标）
    ax2.set_ylabel('Validation Loss (MSE, Log Scale)', color='black')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor='black')

    # 合并图例（包括数据量/策略和线型说明）
    labels = [l.get_label() for l in lines]

    # 创建线型说明的虚拟线条
    from matplotlib.lines import Line2D
    legend_elements = lines + [
        Line2D([0], [0], color='gray', linewidth=lw, linestyle='-', label='Linear Scale (Left Axis)'),
        Line2D([0], [0], color='gray', linewidth=lw * 0.8, linestyle='--', alpha=0.5, label='Log Scale (Right Axis)')
    ]
    legend_labels = labels + ['Linear Scale (Left Axis)', 'Log Scale (Right Axis)']

    fig.legend(legend_elements, legend_labels,
               loc="outside upper center",
               ncol=legend_ncol, frameon=False, fontsize=12)

    # 统一命名格式: ablation_{type}_comparison_curve.png
    filename = f"ablation_{ablation_type}_comparison_curve.png"
    fig.savefig(os.path.join(output_dir, filename), dpi=FIG_DPI)
    plt.close(fig)
    print(f"[保存] {filename}")


# ========================================
# 兼容性包装函数（保持向后兼容）
# ========================================

def plot_freeze_strategy_comparison(
    strategy_results: Dict[str, Tuple[List[float], List[float]]],
    output_dir: str
):
    """
    绘制冻结策略消融对比曲线（兼容性包装）

    Args:
        strategy_results: {strategy_name: (train_losses, val_losses)}
        output_dir: 输出目录
    """
    plot_ablation_comparison_curve(
        ablation_results=strategy_results,
        output_dir=output_dir,
        ablation_type="freeze",
        legend_loc='upper center',
        legend_ncol=5
    )


def plot_transfer_learning_data_size_comparison(
    data_size_results: Dict[int, Tuple[List[float], List[float]]],
    output_dir: str
):
    """
    绘制迁移学习数据量消融对比曲线（兼容性包装）

    Args:
        data_size_results: {n_samples: (train_losses, val_losses)}
        output_dir: 输出目录
    """
    plot_ablation_comparison_curve(
        ablation_results=data_size_results,
        output_dir=output_dir,
        ablation_type="datasize",
        legend_loc='upper center',  # 统一使用正上方
        legend_ncol=6
    )


def plot_from_scratch_data_size_comparison(
    data_size_results: Dict[int, Tuple[List[float], List[float]]],
    output_dir: str
):
    """
    绘制从头训练数据量消融对比曲线（兼容性包装）

    Args:
        data_size_results: {n_samples: (train_losses, val_losses)}
        output_dir: 输出目录
    """
    plot_ablation_comparison_curve(
        ablation_results=data_size_results,
        output_dir=output_dir,
        ablation_type="fromscratch_datasize",
        legend_loc='upper right',
        legend_ncol=6
    )


# ========================================
# 再生器函数
# ========================================

def regenerate_training_curve(
    output_dir: str,
    plot_style: dict = None,
    fig_dpi: int = None
) -> bool:
    """
    从 CSV 再生 training_curve.png

    此函数从已保存的 training_losses.csv 文件重新生成训练曲线图，
    无需重新训练模型。用于在 PNG 被删除时独立恢复。

    Args:
        output_dir: 输出目录（包含 training_losses.csv）
        plot_style: 绘图样式字典（可选，默认使用 PLOT_STYLE）
        fig_dpi: 图像 DPI（可选，默认使用 FIG_DPI）

    Returns:
        True = 成功生成
        False = 生成失败（CSV 不存在）
    """
    csv_path = os.path.join(output_dir, "training_losses.csv")

    if not os.path.exists(csv_path):
        print(f"[警告] CSV 文件不存在，无法再生 PNG: {csv_path}")
        return False

    # 读取损失数据
    losses_df = pd.read_csv(csv_path)
    train_losses = losses_df['train_loss'].tolist()
    val_losses = losses_df['val_loss'].tolist()

    # 使用默认样式
    style = plot_style if plot_style is not None else PLOT_STYLE
    dpi = fig_dpi if fig_dpi is not None else FIG_DPI

    # 重新生成图表
    plot_training_curve(train_losses, val_losses, output_dir, style, dpi)

    print(f"[再生] training_curve.png")
    return True
