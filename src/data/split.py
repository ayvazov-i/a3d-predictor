# TODO: For publication-quality results, replace random split with
# MMseqs2 sequence-identity clustering at 30% identity. Random splits
# overestimate performance on novel proteins.

"""
split.py
--------
Stratified train/val/test split of destress_combined.parquet.
Stratification is by aggrescan3d_avg_value quartile to ensure each
split has a similar target distribution.

Outputs:
  data/splits/train.parquet  (80%)
  data/splits/val.parquet    (10%)
  data/splits/test.parquet   (10%)
"""

from pathlib import Path
import numpy as np
import polars as pl

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
IN_PATH    = ROOT / "data" / "processed" / "destress_combined.parquet"
SPLITS_DIR = ROOT / "data" / "splits"

SEED       = 42
TRAIN_FRAC = 0.80
VAL_FRAC   = 0.10
# TEST_FRAC  = 1 - TRAIN_FRAC - VAL_FRAC  = 0.10


def stratified_split(
    df: pl.DataFrame,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split df into train/val/test stratified by aggrescan3d_avg_value quartile.
    Each quartile stratum is shuffled independently then sliced proportionally,
    so target distribution is balanced across all three splits.
    """
    rng = np.random.default_rng(seed)

    # Assign quartile labels (0–3) using the combined dataset distribution
    values = df["aggrescan3d_avg_value"].to_numpy()
    quantile_edges = np.quantile(values, [0.25, 0.50, 0.75])
    bins = np.digitize(values, quantile_edges)          # 0, 1, 2, or 3

    df = df.with_columns(pl.Series("_stratum", bins))

    train_frames, val_frames, test_frames = [], [], []

    for stratum in range(4):
        stratum_df = df.filter(pl.col("_stratum") == stratum)
        n = len(stratum_df)

        # Shuffle within stratum
        order = rng.permutation(n)
        stratum_df = stratum_df[order]

        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)

        train_frames.append(stratum_df[:n_train])
        val_frames.append(stratum_df[n_train : n_train + n_val])
        test_frames.append(stratum_df[n_train + n_val :])

    train = pl.concat(train_frames).drop("_stratum")
    val   = pl.concat(val_frames).drop("_stratum")
    test  = pl.concat(test_frames).drop("_stratum")

    return train, val, test


def print_split_summary(
    train: pl.DataFrame,
    val: pl.DataFrame,
    test: pl.DataFrame,
    total: int,
) -> None:
    print(f"\n{'='*60}")
    print("SPLIT SUMMARY")
    print(f"{'='*60}")
    header = f"  {'Split':<8}  {'Rows':>8}  {'%Total':>7}  {'A3D mean':>10}  {'A3D std':>9}"
    print(header)
    print(f"  {'-'*54}")
    for name, df in [("train", train), ("val", val), ("test", test)]:
        a3d  = df["aggrescan3d_avg_value"]
        pct  = 100 * len(df) / total
        print(
            f"  {name:<8}  {len(df):>8,}  {pct:>6.1f}%"
            f"  {a3d.mean():>10.4f}  {a3d.std():>9.4f}"
        )
    print(f"  {'─'*54}")
    total_split = len(train) + len(val) + len(test)
    print(f"  {'total':<8}  {total_split:>8,}  {'100.0%':>7}")

    # Per-source breakdown within each split
    print(f"\n  Source breakdown:")
    src_header = f"  {'Split':<8}  {'af2':>8}  {'pdb':>8}"
    print(src_header)
    print(f"  {'-'*28}")
    for name, df in [("train", train), ("val", val), ("test", test)]:
        counts = dict(df.group_by("source").agg(pl.len().alias("n")).iter_rows())
        af2_n  = counts.get("af2", 0)
        pdb_n  = counts.get("pdb", 0)
        print(f"  {name:<8}  {af2_n:>8,}  {pdb_n:>8,}")


def main() -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {IN_PATH} ...")
    df = pl.read_parquet(IN_PATH)
    total = len(df)
    print(f"  Rows loaded: {total:,}")

    print(f"\nSplitting (seed={SEED}, train={TRAIN_FRAC:.0%}, val={VAL_FRAC:.0%}, test=10%) ...")
    train, val, test = stratified_split(df, TRAIN_FRAC, VAL_FRAC, SEED)

    for name, split_df, path in [
        ("train", train, SPLITS_DIR / "train.parquet"),
        ("val",   val,   SPLITS_DIR / "val.parquet"),
        ("test",  test,  SPLITS_DIR / "test.parquet"),
    ]:
        split_df.write_parquet(path, compression="snappy")
        size_mb = path.stat().st_size / 1_048_576
        print(f"  Saved {name:5s}: {len(split_df):>8,} rows → {path.name}  ({size_mb:.1f} MB)")

    print_split_summary(train, val, test, total)
    print(f"\nDone. Splits written to {SPLITS_DIR}/")


if __name__ == "__main__":
    main()
