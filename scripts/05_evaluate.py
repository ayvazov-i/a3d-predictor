"""
05_evaluate.py
--------------
Final evaluation of all trained models on the held-out test set.

Models evaluated:
  • Ridge (ESM-2)    — loaded from models_saved/ridge_esm.pkl
  • XGBoost (ESM-2)  — loaded from models_saved/xgb_esm.json
  • MLP (ESM-2)      — loaded from models_saved/best_mlp.pt  (skipped if missing)

Outputs (all written to results/):
  predicted_vs_actual.png
  residual_plot.png
  error_distribution.png
  performance_by_length.png
  performance_by_source.png
  final_results.csv
"""

import pickle
import sys
import warnings
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from models.mlp import MLPRegressor

SPLITS_DIR  = ROOT / "data" / "splits"
EMB_DIR     = ROOT / "data" / "processed"
MODELS_DIR  = ROOT / "models_saved"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "aggrescan3d_avg_value"

# Plot style
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

LENGTH_BINS   = [0, 100, 300, 500, 1000, float("inf")]
LENGTH_LABELS = ["0–100", "100–300", "300–500", "500–1000", "1000+"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_test_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Returns:
        X          : float32 [N, 1280] embeddings
        y          : float32 [N] labels
        meta_df    : DataFrame with design_name, seq_length, source aligned to X/y
    """
    h5_path = EMB_DIR / "embeddings_test.h5"
    pq_path = SPLITS_DIR / "test.parquet"

    print(f"Loading test embeddings from {h5_path.name} ...", end=" ", flush=True)
    with h5py.File(h5_path, "r") as f:
        embeddings   = f["embeddings"][:]
        design_names = np.array([
            n.decode() if isinstance(n, bytes) else n
            for n in f["design_names"][:]
        ])
    print(f"{len(design_names):,} sequences")

    print(f"Loading test labels from {pq_path.name} ...", end=" ", flush=True)
    df = pd.read_parquet(pq_path, columns=["design_name", TARGET_COL, "seq_length", "source"])
    print(f"{len(df):,} rows  (skipped {len(df) - len(design_names):,} with seq_len > 1022)")

    meta = df.set_index("design_name").loc[design_names].reset_index()
    y    = meta[TARGET_COL].to_numpy(dtype=np.float32)

    return embeddings, y, meta


# ── Metrics ───────────────────────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r2   = float(r2_score(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae  = float(mean_absolute_error(y_true, y_pred))
        pr   = float(pearsonr(y_true, y_pred).statistic)
        sr   = float(spearmanr(y_true, y_pred).statistic)
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "Pearson_r": pr, "Spearman_rho": sr}


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_ridge(X: np.ndarray) -> np.ndarray:
    path = MODELS_DIR / "ridge_esm.pkl"
    with open(path, "rb") as fh:
        bundle = pickle.load(fh)
    X_sc = bundle["scaler"].transform(X)
    return bundle["model"].predict(X_sc).astype(np.float32)


def predict_xgb(X: np.ndarray) -> np.ndarray:
    path = MODELS_DIR / "xgb_esm.json"
    xgb  = XGBRegressor()
    xgb.load_model(path)
    return xgb.predict(X).astype(np.float32)


def predict_mlp(X: np.ndarray) -> np.ndarray | None:
    path = MODELS_DIR / "best_mlp.pt"
    if not path.exists():
        print(f"  [SKIP] {path.name} not found — MLP not evaluated.")
        return None
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt      = torch.load(path, map_location=device)
    # Support both raw state_dict and dict with metadata
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    cfg        = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model = MLPRegressor(
        input_dim   = cfg.get("input_dim",   1280),
        hidden_dims = cfg.get("hidden_dims", (512, 256, 128)),
        dropout     = cfg.get("dropout",     0.0),   # no dropout at inference
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    tensor = torch.from_numpy(X).to(device)
    preds  = []
    bs     = 4096
    with torch.no_grad():
        for i in range(0, len(X), bs):
            preds.append(model(tensor[i : i + bs]).cpu().numpy())
    return np.concatenate(preds).astype(np.float32)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                              model_name: str, path: Path) -> None:
    metrics = evaluate(y_true, y_pred)
    vmin, vmax = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    margin = (vmax - vmin) * 0.05

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hexbin(y_true, y_pred, gridsize=80, cmap="viridis", mincnt=1, linewidths=0)
    lim = [vmin - margin, vmax + margin]
    ax.plot(lim, lim, "r--", lw=1.2, label="Identity")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual aggrescan3d_avg_value")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{model_name} — Predicted vs Actual (test set)")
    ax.text(0.05, 0.93,
            f"R²={metrics['R2']:.4f}  Pearson={metrics['Pearson_r']:.4f}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax.legend(fontsize=9)
    plt.colorbar(ax.collections[0], ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path.name}")


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                   model_name: str, path: Path) -> None:
    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hexbin(y_pred, residuals, gridsize=80, cmap="RdYlBu_r", mincnt=1, linewidths=0)
    ax.axhline(0, color="black", lw=1.0, ls="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (pred − actual)")
    ax.set_title(f"{model_name} — Residuals (test set)")
    plt.colorbar(ax.collections[0], ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path.name}")


def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray,
                             model_name: str, path: Path) -> None:
    errors = y_pred - y_true
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(errors, bins=120, color="#4C72B0", edgecolor="none", alpha=0.85)
    ax.axvline(0, color="red", lw=1.2, ls="--")
    ax.axvline(errors.mean(), color="orange", lw=1.2, ls="--",
               label=f"mean={errors.mean():.4f}")
    ax.set_xlabel("Prediction error (pred − actual)")
    ax.set_ylabel("Count")
    ax.set_title(f"{model_name} — Error Distribution (test set)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path.name}")


def plot_performance_by_length(y_true: np.ndarray, y_pred: np.ndarray,
                                seq_lengths: np.ndarray,
                                model_name: str, path: Path) -> None:
    bucket = pd.cut(seq_lengths, bins=LENGTH_BINS, labels=LENGTH_LABELS, right=False)
    records = []
    for label in LENGTH_LABELS:
        mask = bucket == label
        if mask.sum() < 10:
            continue
        m = evaluate(y_true[mask], y_pred[mask])
        records.append({"bucket": label, "n": int(mask.sum()), **m})
    df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    colors = plt.cm.tab10.colors
    for ax, col, title in zip(axes,
                               ["R2", "RMSE", "Pearson_r"],
                               ["R²", "RMSE", "Pearson r"]):
        bars = ax.bar(df["bucket"], df[col], color=colors[:len(df)], edgecolor="none")
        ax.set_xlabel("Sequence length bucket")
        ax.set_ylabel(title)
        ax.set_title(f"{title} by length")
        ax.tick_params(axis="x", rotation=30)
        for bar, n in zip(bars, df["n"]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"n={n:,}", ha="center", va="bottom", fontsize=7)
    fig.suptitle(f"{model_name} — Performance by Sequence Length (test set)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path.name}")


def plot_performance_by_source(y_true: np.ndarray, y_pred: np.ndarray,
                                sources: np.ndarray,
                                model_name: str, path: Path) -> None:
    unique_sources = sorted(set(sources))
    records = []
    for src in unique_sources:
        mask = sources == src
        m    = evaluate(y_true[mask], y_pred[mask])
        records.append({"source": src, "n": int(mask.sum()), **m})
    df = pd.DataFrame(records)

    metrics_to_plot = ["R2", "RMSE", "MAE", "Pearson_r", "Spearman_rho"]
    titles          = ["R²", "RMSE", "MAE", "Pearson r", "Spearman ρ"]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(14, 4))
    x    = np.arange(len(df))
    for ax, col, title in zip(axes, metrics_to_plot, titles):
        bars = ax.bar(x, df[col], color=["#4C72B0", "#DD8452"][:len(df)], edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{row['source']}\n(n={row['n']:,})" for _, row in df.iterrows()],
            fontsize=9,
        )
        ax.set_ylabel(title)
        ax.set_title(title)
        for bar, val in zip(bars, df[col]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(bar.get_height()) * 0.01,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"{model_name} — Performance by Source (test set)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 70)
    print("05_evaluate.py — Final test-set evaluation")
    print("=" * 70)

    # ── Load test data ────────────────────────────────────────────────────────
    print("\n--- Loading test data ---")
    X_test, y_test, meta = load_test_data()
    seq_lengths = meta["seq_length"].to_numpy()
    sources     = meta["source"].to_numpy()

    # ── Run all models ────────────────────────────────────────────────────────
    model_preds: dict[str, np.ndarray] = {}

    print("\n--- Ridge (ESM-2) ---")
    model_preds["Ridge (ESM-2)"] = predict_ridge(X_test)

    print("\n--- XGBoost (ESM-2) ---")
    model_preds["XGBoost (ESM-2)"] = predict_xgb(X_test)

    print("\n--- MLP (ESM-2) ---")
    mlp_preds = predict_mlp(X_test)
    if mlp_preds is not None:
        model_preds["MLP (ESM-2)"] = mlp_preds

    # ── Plots (use best available model = lowest RMSE) ────────────────────────
    test_metrics = {
        name: evaluate(y_test, preds)
        for name, preds in model_preds.items()
    }
    best_name = min(test_metrics, key=lambda n: test_metrics[n]["RMSE"])
    best_pred = model_preds[best_name]
    print(f"\n--- Generating plots (using best model: {best_name}) ---")

    plot_predicted_vs_actual(
        y_test, best_pred, best_name,
        RESULTS_DIR / "predicted_vs_actual.png",
    )
    plot_residuals(
        y_test, best_pred, best_name,
        RESULTS_DIR / "residual_plot.png",
    )
    plot_error_distribution(
        y_test, best_pred, best_name,
        RESULTS_DIR / "error_distribution.png",
    )
    plot_performance_by_length(
        y_test, best_pred, seq_lengths, best_name,
        RESULTS_DIR / "performance_by_length.png",
    )
    plot_performance_by_source(
        y_test, best_pred, sources, best_name,
        RESULTS_DIR / "performance_by_source.png",
    )

    # ── Comparison table ──────────────────────────────────────────────────────
    # Merge with any baseline_results (val-set) rows for context
    baseline_path = RESULTS_DIR / "baseline_results.csv"
    if baseline_path.exists():
        val_df = pd.read_csv(baseline_path)
        val_df.insert(0, "split", "val")
        val_df.insert(1, "source_file", "baseline_results.csv")
    else:
        val_df = pd.DataFrame()

    test_rows = [
        {"split": "test", "source_file": "05_evaluate.py", "model": name, **m}
        for name, m in test_metrics.items()
    ]
    test_df = pd.DataFrame(test_rows)

    print(f"\n{'='*80}")
    print("FINAL TEST-SET RESULTS")
    print(f"{'='*80}")
    print(f"  {'Model':<24s}  {'R²':>8}  {'RMSE':>8}  {'MAE':>8}  {'Pearson':>8}  {'Spearman':>9}")
    print(f"  {'-'*74}")
    for _, row in test_df.iterrows():
        print(
            f"  {row['model']:<24s}  {row['R2']:>8.4f}  {row['RMSE']:>8.4f}"
            f"  {row['MAE']:>8.4f}  {row['Pearson_r']:>8.4f}"
            f"  {row['Spearman_rho']:>9.4f}"
        )

    if not val_df.empty:
        print(f"\n  {'─'*74}")
        print("  (Validation-set baselines for context)")
        print(f"  {'-'*74}")
        for _, row in val_df.iterrows():
            print(
                f"  {row['model']:<24s}  {row['R2']:>8.4f}  {row['RMSE']:>8.4f}"
                f"  {row['MAE']:>8.4f}  {row['Pearson_r']:>8.4f}"
                f"  {row['Spearman_rho']:>9.4f}  [val]"
            )
    print(f"  {'='*74}")

    # ── Save final results ────────────────────────────────────────────────────
    final_path = RESULTS_DIR / "final_results.csv"
    combined = pd.concat([val_df, test_df], ignore_index=True) if not val_df.empty else test_df
    combined.to_csv(final_path, index=False)
    print(f"\nFinal results saved → {final_path}")


if __name__ == "__main__":
    main()
