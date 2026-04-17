"""
03b_train_baseline_esm.py
--------------------------
Train and evaluate baseline models (Ridge, XGBoost) on ESM-2 1280-dim
embeddings to predict aggrescan3d_avg_value.

Results are merged into results/baseline_results.csv alongside any
previously saved hand-crafted-feature baselines.
"""

import sys
import time
from pathlib import Path

import pickle

import h5py
import numpy as np
import polars as pl
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

SPLITS_DIR   = ROOT / "data" / "splits"
EMB_DIR      = ROOT / "data" / "processed"
RESULTS_DIR  = ROOT / "results"
MODELS_DIR   = ROOT / "models_saved"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "aggrescan3d_avg_value"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_embeddings_and_labels(split: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load ESM-2 embeddings from HDF5 and align labels from parquet.
    Returns (X, y) where rows are matched by design_name.
    Sequences that were skipped during embedding extraction (>1022 residues)
    are absent from the HDF5 and therefore excluded from both X and y.
    """
    h5_path  = EMB_DIR / f"embeddings_{split}.h5"
    pq_path  = SPLITS_DIR / f"{split}.parquet"

    print(f"  [{split}] Loading embeddings from {h5_path.name} ...", end=" ", flush=True)
    with h5py.File(h5_path, "r") as f:
        embeddings   = f["embeddings"][:]              # [N, 1280] float32
        design_names = np.array([
            n.decode() if isinstance(n, bytes) else n
            for n in f["design_names"][:]
        ])
    print(f"{len(design_names):,} sequences, shape {embeddings.shape}")

    print(f"  [{split}] Loading labels from {pq_path.name} ...", end=" ", flush=True)
    df = pl.read_parquet(pq_path, columns=["design_name", TARGET_COL])
    print(f"{len(df):,} rows")

    # Build a name→label lookup and align to HDF5 order
    label_map = dict(zip(df["design_name"].to_list(), df[TARGET_COL].to_list()))
    labels = np.array([label_map[n] for n in design_names], dtype=np.float32)

    return embeddings, labels


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    r2   = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    pr   = float(pearsonr(y_true, y_pred).statistic)
    sr   = float(spearmanr(y_true, y_pred).statistic)
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "Pearson_r": pr, "Spearman_rho": sr}


def print_metrics(name: str, metrics: dict) -> None:
    print(
        f"  {name:<22s}  R²={metrics['R2']:+.4f}  RMSE={metrics['RMSE']:.4f}"
        f"  MAE={metrics['MAE']:.4f}  Pearson={metrics['Pearson_r']:.4f}"
        f"  Spearman={metrics['Spearman_rho']:.4f}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    use_gpu = torch.cuda.is_available()
    print(f"\nGPU available: {use_gpu}")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n=== Loading embeddings & labels ===")
    X_train, y_train = load_embeddings_and_labels("train")
    X_val,   y_val   = load_embeddings_and_labels("val")

    print(f"\n  Train: X={X_train.shape}  y={y_train.shape}  "
          f"y_mean={y_train.mean():.4f}  y_std={y_train.std():.4f}")
    print(f"  Val:   X={X_val.shape}  y={y_val.shape}  "
          f"y_mean={y_val.mean():.4f}  y_std={y_val.std():.4f}")

    # ── 2. Ridge on ESM-2 embeddings ─────────────────────────────────────────
    print("\n=== Model: Ridge (ESM-2) ===")
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_va_sc = scaler.transform(X_val)

    t0 = time.time()
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr_sc, y_train)
    elapsed = time.time() - t0
    print(f"  Trained in {elapsed:.1f}s")

    ridge_preds   = ridge.predict(X_va_sc).astype(np.float32)
    ridge_metrics = evaluate(y_val, ridge_preds)
    print_metrics("Ridge (ESM-2)", ridge_metrics)

    ridge_save = MODELS_DIR / "ridge_esm.pkl"
    with open(ridge_save, "wb") as fh:
        pickle.dump({"model": ridge, "scaler": scaler}, fh)
    print(f"  Saved → {ridge_save}")

    # ── 3. XGBoost on ESM-2 embeddings ───────────────────────────────────────
    print("\n=== Model: XGBoost (ESM-2) ===")
    xgb_params = dict(
        max_depth             = 6,
        n_estimators          = 500,
        learning_rate         = 0.05,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        tree_method           = "hist",
        device                = "cuda" if use_gpu else "cpu",
        n_jobs                = -1,
        random_state          = 42,
        early_stopping_rounds = 20,
        eval_metric           = "rmse",
    )
    print(f"  device={'cuda' if use_gpu else 'cpu'}  max_depth=6  n_estimators=500  "
          f"early_stopping=20")

    xgb = XGBRegressor(**xgb_params)
    t0  = time.time()
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    elapsed    = time.time() - t0
    best_round = xgb.best_iteration
    print(f"  Trained in {elapsed:.1f}s  |  best round: {best_round}")

    xgb_preds   = xgb.predict(X_val).astype(np.float32)
    xgb_metrics = evaluate(y_val, xgb_preds)
    print_metrics("XGBoost (ESM-2)", xgb_metrics)

    xgb_save = MODELS_DIR / "xgb_esm.json"
    xgb.save_model(xgb_save)
    print(f"  Saved → {xgb_save}")

    # ── 4. Comparison table ───────────────────────────────────────────────────
    new_rows = [
        {"model": "Ridge (ESM-2)",   **ridge_metrics},
        {"model": "XGBoost (ESM-2)", **xgb_metrics},
    ]

    results_path = RESULTS_DIR / "baseline_results.csv"
    if results_path.exists():
        existing = pl.read_csv(results_path)
        # Drop any stale ESM rows so re-running is idempotent
        existing = existing.filter(
            ~pl.col("model").str.contains("ESM-2")
        )
        combined = pl.concat([existing, pl.DataFrame(new_rows)])
    else:
        combined = pl.DataFrame(new_rows)

    print(f"\n{'='*80}")
    print("VALIDATION SET COMPARISON — ALL MODELS")
    print(f"{'='*80}")
    print(f"  {'Model':<24s}  {'R²':>8}  {'RMSE':>8}  {'MAE':>8}  "
          f"{'Pearson':>8}  {'Spearman':>9}")
    print(f"  {'-'*72}")
    for row in combined.iter_rows(named=True):
        print(
            f"  {row['model']:<24s}  {row['R2']:>8.4f}  {row['RMSE']:>8.4f}"
            f"  {row['MAE']:>8.4f}  {row['Pearson_r']:>8.4f}"
            f"  {row['Spearman_rho']:>9.4f}"
        )
    print(f"  {'─'*72}")

    combined.write_csv(results_path)
    print(f"\nResults saved → {results_path}")


if __name__ == "__main__":
    main()
