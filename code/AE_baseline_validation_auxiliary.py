"""
AE_baseline_validation_auxiliary.py
====================================
Validate the pretrain AE model's damage detection on the source domain.
Tests healthy vs damaged samples without domain shift.
Called via run_baseline_validation.py.

Usage:
    cd script && python run_baseline_validation.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = os.path.join(SCRIPT_DIR, "AE_model_train_and_detect_output")
PREPROCESS_BASE = os.path.join(SCRIPT_DIR, "AD_preprocess_datasets_output")

PRETRAIN_PTH = os.path.join(
    OUTPUT_BASE, "Damage_Repaired", "pretrain", "autoencoder.pth"
)
HEALTH_NPZ = os.path.join(
    PREPROCESS_BASE, "health_original_2000", "preprocessed_data_raw.npz"
)
DAMAGE_NPZ = os.path.join(
    PREPROCESS_BASE, "first_damage_original_100", "preprocessed_data_raw.npz"
)

OUTPUT_DIR = os.path.join(OUTPUT_BASE, "baseline_validation")

# ---------------------------------------------------------------------------
# Model architecture (must match pretrain config)
# ---------------------------------------------------------------------------
AE_CONFIG = dict(
    encoder_dims=[768, 384, 192],
    latent_dim=192,
    decoder_dims=[192, 384, 768],
    dropout=0.0,
    activation="relu",
)

# ---------------------------------------------------------------------------
# Plot style (academic conventions from CLAUDE.md)
# ---------------------------------------------------------------------------
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
FIG_DPI = 300

# Morandi palette
COLOR_HEALTHY = "#8FAADC"   # soft blue-grey
COLOR_DAMAGED = "#C4756E"   # soft terracotta


def apply_style():
    plt.rcParams.update(PLOT_STYLE)


# ---------------------------------------------------------------------------
# Import project modules
# ---------------------------------------------------------------------------
sys.path.insert(0, SCRIPT_DIR)
from AE_train_model_auxiliary import Autoencoder  # noqa: E402
from AE_model_train_and_detect_auxiliary import (  # noqa: E402
    compute_anomaly_scores,
    compute_detection_metrics,
)


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 60)
    print("Baseline Validation: Pretrain AE on Source Domain")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Check files
    # ------------------------------------------------------------------
    if not os.path.exists(PRETRAIN_PTH):
        print(f"[ERROR] Pretrain model not found: {PRETRAIN_PTH}")
        sys.exit(1)

    if not os.path.exists(HEALTH_NPZ):
        print(f"[ERROR] Health data not found: {HEALTH_NPZ}")
        sys.exit(1)

    if not os.path.exists(DAMAGE_NPZ):
        print(
            f"[WARNING] Damage data not found: {DAMAGE_NPZ}\n"
            f"  The file first_damage_original_100/preprocessed_data_raw.npz "
            f"does not exist yet.\n"
            f"  Please run the preprocessing pipeline first:\n"
            f"    python AD_preprocess_datasets.py\n"
            f"  Exiting."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ------------------------------------------------------------------
    # 3. Load pretrain model
    # ------------------------------------------------------------------
    print(f"\n[Model] Loading pretrain model ...")
    # Determine input_dim from health data
    health_data = np.load(HEALTH_NPZ)
    V_health_all = health_data["V"].astype(np.float32)
    input_dim = V_health_all.shape[1]
    print(f"  Input dim: {input_dim}")

    model = Autoencoder(
        input_dim,
        AE_CONFIG["encoder_dims"],
        AE_CONFIG["latent_dim"],
        AE_CONFIG["decoder_dims"],
        AE_CONFIG["dropout"],
        AE_CONFIG["activation"],
    ).to(device)
    model.load_state_dict(torch.load(PRETRAIN_PTH, map_location=device))
    model.eval()
    print(f"  Loaded: {PRETRAIN_PTH}")

    # ------------------------------------------------------------------
    # 4. Load data
    # ------------------------------------------------------------------
    # Healthy: last 200 samples of health_original_2000 as validation portion
    V_healthy = V_health_all[-200:]
    print(f"\n[Data] Healthy control (last 200 of health_original_2000): {V_healthy.shape}")

    # Damaged: full first_damage_original_100
    damage_data = np.load(DAMAGE_NPZ)
    V_damaged = damage_data["V"].astype(np.float32)
    print(f"[Data] Damaged test (first_damage_original_100): {V_damaged.shape}")

    # ------------------------------------------------------------------
    # 5. Compute anomaly scores (mean-channel + max-channel)
    # ------------------------------------------------------------------
    print(f"\n[Inference] Computing anomaly scores ...")
    scores_healthy = compute_anomaly_scores(V_healthy, model, device, method='mean')
    scores_damaged = compute_anomaly_scores(V_damaged, model, device, method='mean')
    scores_healthy_max = compute_anomaly_scores(V_healthy, model, device, method='max')
    scores_damaged_max = compute_anomaly_scores(V_damaged, model, device, method='max')

    print(f"  [Mean-ch] Healthy: mean={scores_healthy.mean():.6f}, "
          f"std={scores_healthy.std():.6f}, "
          f"range=[{scores_healthy.min():.6f}, {scores_healthy.max():.6f}]")
    print(f"  [Mean-ch] Damaged: mean={scores_damaged.mean():.6f}, "
          f"std={scores_damaged.std():.6f}, "
          f"range=[{scores_damaged.min():.6f}, {scores_damaged.max():.6f}]")
    print(f"  [Max-ch]  Healthy: mean={scores_healthy_max.mean():.6f}, "
          f"std={scores_healthy_max.std():.6f}, "
          f"range=[{scores_healthy_max.min():.6f}, {scores_healthy_max.max():.6f}]")
    print(f"  [Max-ch]  Damaged: mean={scores_damaged_max.mean():.6f}, "
          f"std={scores_damaged_max.std():.6f}, "
          f"range=[{scores_damaged_max.min():.6f}, {scores_damaged_max.max():.6f}]")

    # ------------------------------------------------------------------
    # 6. Compute detection metrics (both scoring methods)
    # ------------------------------------------------------------------
    print(f"\n[Metrics] Computing detection metrics ...")
    metrics = compute_detection_metrics(scores_damaged, scores_healthy)
    metrics_max = compute_detection_metrics(scores_damaged_max, scores_healthy_max)

    print(f"  [Mean-channel] AUC={metrics['auc']:.4f}, "
          f"TPR@FPR5%={metrics['tpr_at_fpr']:.4f}, "
          f"Best F1={metrics['best_f1']:.4f}")
    print(f"  [Max-channel]  AUC={metrics_max['auc']:.4f}, "
          f"TPR@FPR5%={metrics_max['tpr_at_fpr']:.4f}, "
          f"Best F1={metrics_max['best_f1']:.4f}")

    # ------------------------------------------------------------------
    # 7. Save metrics CSV (both scoring methods)
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path = os.path.join(OUTPUT_DIR, "baseline_detection_metrics.csv")
    df_metrics = pd.DataFrame([{
        "Model": "Pretrain (source domain)",
        "AUC": round(metrics["auc"], 4),
        "TPR@FPR5%": round(metrics["tpr_at_fpr"], 4),
        "Best_F1": round(metrics["best_f1"], 4),
    }])
    df_metrics.to_csv(csv_path, index=False)
    print(f"\n[Saved] {csv_path}")

    csv_path_max = os.path.join(OUTPUT_DIR, "baseline_detection_metrics_max.csv")
    df_metrics_max = pd.DataFrame([{
        "Model": "Pretrain (source domain)",
        "AUC": round(metrics_max["auc"], 4),
        "TPR@FPR5%": round(metrics_max["tpr_at_fpr"], 4),
        "Best_F1": round(metrics_max["best_f1"], 4),
    }])
    df_metrics_max.to_csv(csv_path_max, index=False)
    print(f"[Saved] {csv_path_max}")

    # ------------------------------------------------------------------
    # 8. Plot fig_baseline_dist.png — anomaly score distribution
    # ------------------------------------------------------------------
    apply_style()

    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)

    # Use KDE curves for cleaner visualization
    from sklearn.neighbors import KernelDensity

    def _kde(data, n_points=512):
        """Compute KDE density for 1-D data."""
        clean = data[np.isfinite(data)]
        std = clean.std(ddof=1)
        if std == 0 or not np.isfinite(std):
            std = 1e-3
        bw = 1.06 * std * len(clean) ** (-1.0 / 5.0)
        if bw <= 0 or not np.isfinite(bw):
            bw = 1e-3
        lo, hi = max(0, clean.min() - 3 * std), clean.max() + 3 * std
        xs = np.linspace(lo, hi, n_points)
        kde = KernelDensity(kernel="gaussian", bandwidth=bw)
        kde.fit(clean[:, None])
        density = np.exp(kde.score_samples(xs[:, None]))
        return xs, density

    xs_h, dens_h = _kde(scores_healthy)
    xs_d, dens_d = _kde(scores_damaged)

    ax.fill_between(xs_h, dens_h, alpha=0.35, color=COLOR_HEALTHY)
    ax.plot(xs_h, dens_h, color=COLOR_HEALTHY, lw=1.5, label="Healthy (source domain)")
    ax.fill_between(xs_d, dens_d, alpha=0.35, color=COLOR_DAMAGED)
    ax.plot(xs_d, dens_d, color=COLOR_DAMAGED, lw=1.5, label="Damaged (source domain)")

    ax.set_xlabel("Per-sample MAE (anomaly score)")
    ax.set_ylabel("Density")
    ax.tick_params(direction="in")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="outside upper center",
               ncol=len(labels), frameon=False, fontsize=12)

    dist_path = os.path.join(OUTPUT_DIR, "fig_baseline_dist.png")
    fig.savefig(dist_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"[Saved] {dist_path}")

    # ------------------------------------------------------------------
    # 9. Plot fig_baseline_roc.png — ROC curve
    # ------------------------------------------------------------------
    apply_style()

    fig, ax = plt.subplots(figsize=(6.5, 7.3), constrained_layout=True)

    roc_color = "#7B6D8D"  # Morandi muted purple
    ax.plot(
        metrics["fprs"],
        metrics["tprs"],
        color=roc_color,
        lw=1.5,
        label=f"Pretrain (AUC = {metrics['auc']:.3f})",
    )
    ax.plot([0, 1], [0, 1], color="#cccccc", lw=0.8, linestyle="--")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(direction="in")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="outside upper center",
               ncol=1, frameon=False, fontsize=12)

    roc_path = os.path.join(OUTPUT_DIR, "fig_baseline_roc.png")
    fig.savefig(roc_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"[Saved] {roc_path}")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Baseline validation complete. Outputs in:")
    print(f"  {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
