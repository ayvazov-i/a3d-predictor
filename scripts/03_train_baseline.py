"""
03_train_baseline.py
--------------------
Train and evaluate baseline models (Ridge, XGBoost) on hand-crafted
sequence features to predict aggrescan3d_avg_value.

Results saved to results/baseline_results.csv
"""

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features.sequence_features import FEATURE_NAMES, sequences_to_features

SPLITS_DIR  = ROOT / "data" / "splits"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_split(name: str) -> tuple[list[str], np.ndarray]:
    path = SPLITS_DIR / f"{name}.parquet"
    print(f"  Loading {path.name} ...", end=" ", flush=True)
    df = pl.read_parquet(path, columns=["full_sequence", "aggrescan3d_avg_value"])
    seqs   = df["full_sequence"].to_list()
    labels = df["aggrescan3d_avg_value"].to_numpy(allow_copy=True).astype(np.float32)
    print(f"{len(seqs):,} rows")
    return seqs, labels


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    r2   = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    pr   = float(pearsonr(y_true, y_pred).statistic)
    sr   = float(spearmanr(y_true, y_pred).statistic)
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "Pearson_r": pr, "Spearman_rho": sr}


def print_metrics(name: str, metrics: dict) -> None:
    print(
        f"  {name:<12s}  R²={metrics['R2']:+.4f}  RMSE={metrics['RMSE']:.4f}"
        f"  MAE={metrics['MAE']:.4f}  Pearson={metrics['Pearson_r']:.4f}"
        f"  Spearman={metrics['Spearman_rho']:.4f}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── 1. Load splits ────────────────────────────────────────────────────────
    print("\n=== Loading data ===")
    train_seqs, y_train = load_split("train")
    val_seqs,   y_val   = load_split("val")

    # ── 2. Compute features ───────────────────────────────────────────────────
    print("\n=== Computing hand-crafted features ===")
    t0 = time.time()
    X_train = sequences_to_features(train_seqs, show_progress=True)
    print(f"  Train features: {X_train.shape}  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    X_val = sequences_to_features(val_seqs, show_progress=True)
    print(f"  Val   features: {X_val.shape}  ({time.time()-t0:.1f}s)")

    # ── 3a. Ridge regression ──────────────────────────────────────────────────
    print("\n=== Model A: Ridge Regression ===")
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_va_sc = scaler.transform(X_val)

    t0 = time.time()
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr_sc, y_train)
    print(f"  Trained in {time.time()-t0:.1f}s")

    ridge_preds  = ridge.predict(X_va_sc).astype(np.float32)
    ridge_metrics = evaluate(y_val, ridge_preds)
    print_metrics("Ridge", ridge_metrics)

    # ── 3b. XGBoost ───────────────────────────────────────────────────────────
    print("\n=== Model B: XGBoost ===")
    xgb = XGBRegressor(
        max_depth        = 6,
        n_estimators     = 500,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        tree_method      = "hist",
        device           = "cpu",
        n_jobs           = -1,
        random_state     = 42,
        early_stopping_rounds = 20,
        eval_metric      = "rmse",
    )
    t0 = time.time()
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    elapsed = time.time() - t0
    best_round = xgb.best_iteration
    print(f"  Trained in {elapsed:.1f}s  |  best round: {best_round}")

    xgb_preds   = xgb.predict(X_val).astype(np.float32)
    xgb_metrics = evaluate(y_val, xgb_preds)
    print_metrics("XGBoost", xgb_metrics)

    # ── 4. Comparison table ───────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("VALIDATION SET COMPARISON")
    print(f"{'='*72}")
    print(f"  {'Model':<14s}  {'R²':>7}  {'RMSE':>7}  {'MAE':>7}  {'Pearson':>8}  {'Spearman':>9}")
    print(f"  {'-'*62}")
    results_rows = []
    for name, metrics in [("Ridge", ridge_metrics), ("XGBoost", xgb_metrics)]:
        print(
            f"  {name:<14s}  {metrics['R2']:>7.4f}  {metrics['RMSE']:>7.4f}"
            f"  {metrics['MAE']:>7.4f}  {metrics['Pearson_r']:>8.4f}"
            f"  {metrics['Spearman_rho']:>9.4f}"
        )
        results_rows.append({"model": name, **metrics})
    print(f"  {'─'*62}")

    # ── 5. Save results ───────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "baseline_results.csv"
    pl.DataFrame(results_rows).write_csv(out_path)
    print(f"\nResults saved to {out_path}")

    # Feature importances (top 10 for XGBoost)
    importances = xgb.feature_importances_
    top10_idx   = np.argsort(importances)[::-1][:10]
    print(f"\nTop 10 XGBoost feature importances:")
    for rank, idx in enumerate(top10_idx, 1):
        print(f"  {rank:2d}. {FEATURE_NAMES[idx]:<30s}  {importances[idx]:.4f}")


if __name__ == "__main__":
    main()
