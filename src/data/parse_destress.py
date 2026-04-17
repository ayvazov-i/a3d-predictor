"""
parse_destress.py
-----------------
Load, clean, and merge the DE-STRESS AF2 and PDB datasets.
Outputs: data/processed/destress_combined.parquet
"""

from pathlib import Path
import polars as pl

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
RAW_DIR    = ROOT / "data" / "raw"
PROC_DIR   = ROOT / "data" / "processed"
OUT_PATH   = PROC_DIR / "destress_combined.parquet"

AF2_CSV    = RAW_DIR / "destress_data_af2.csv"
PDB_CSV    = RAW_DIR / "destress_data_pdb_082024.csv"

KEEP_COLS = [
    "design_name",
    "full_sequence",
    "seq_length",           # derived
    "source",               # derived
    "aggrescan3d_total_value",
    "aggrescan3d_avg_value",
    "aggrescan3d_min_value",
    "aggrescan3d_max_value",
    "num_residues",
    "hydrophobic_fitness",
    "isoelectric_point",
    "charge",
    "mass",
    "packing_density",
    "budeff_total",
    "evoef2_total",
    "dfire2_total",
    "rosetta_total",
]


def process_dataset(csv_path: Path, source_label: str) -> pl.DataFrame:
    print(f"\n{'─'*60}")
    print(f"Loading {csv_path.name} ...")
    df = pl.read_csv(csv_path, infer_schema_length=10_000)
    print(f"  Raw shape          : {df.shape[0]:>8,} rows × {df.shape[1]} cols")

    # Cast columns that may have been read as String due to sentinel values
    for col in ["evoef2_total", "evoef2_intraR_total"]:
        if col in df.columns and df[col].dtype == pl.String:
            df = df.with_columns(
                pl.col(col).cast(pl.Float64, strict=False)
            )

    # a. Drop rows missing the target or the sequence
    df = df.filter(
        pl.col("aggrescan3d_avg_value").is_not_null()
        & pl.col("full_sequence").is_not_null()
    )
    print(f"  After null filter  : {df.shape[0]:>8,} rows")

    # b. Remove duplicate design_name (keep first occurrence)
    before = df.shape[0]
    df = df.unique(subset=["design_name"], keep="first", maintain_order=True)
    dropped = before - df.shape[0]
    print(f"  After dedup        : {df.shape[0]:>8,} rows  (dropped {dropped:,} dupes)")

    # c. Add source column
    df = df.with_columns(pl.lit(source_label).alias("source"))

    # d. Add seq_length column
    df = df.with_columns(
        pl.col("full_sequence").str.len_chars().alias("seq_length")
    )

    # e. Select final columns (only those present in this file)
    present = [c for c in KEEP_COLS if c in df.columns]
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        print(f"  WARNING – columns not found, will be null: {missing}")
        for c in missing:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(c))
    df = df.select(KEEP_COLS)

    return df


def print_summary(df: pl.DataFrame) -> None:
    print(f"\n{'='*60}")
    print("COMBINED DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Total rows         : {df.shape[0]:>8,}")
    print(f"  Total columns      : {df.shape[1]:>8,}")

    print(f"\n  Rows per source:")
    for row in df.group_by("source").agg(pl.len().alias("n")).sort("source").iter_rows():
        print(f"    {row[0]:<6s}: {row[1]:>8,}")

    a3d = df["aggrescan3d_avg_value"].drop_nulls()
    print(f"\n  aggrescan3d_avg_value stats (n={a3d.len():,}):")
    print(f"    mean   : {a3d.mean():>10.4f}")
    print(f"    std    : {a3d.std():>10.4f}")
    print(f"    min    : {a3d.min():>10.4f}")
    print(f"    25%    : {a3d.quantile(0.25):>10.4f}")
    print(f"    50%    : {a3d.quantile(0.50):>10.4f}")
    print(f"    75%    : {a3d.quantile(0.75):>10.4f}")
    print(f"    max    : {a3d.max():>10.4f}")

    sl = df["seq_length"].drop_nulls()
    print(f"\n  seq_length stats (n={sl.len():,}):")
    print(f"    min    : {sl.min():>8,}")
    print(f"    max    : {sl.max():>8,}")
    print(f"    mean   : {sl.mean():>10.1f}")
    print(f"    median : {sl.median():>10.1f}")

    null_counts = {c: df[c].null_count() for c in KEEP_COLS if c not in ("design_name", "full_sequence", "source", "seq_length")}
    any_nulls = {c: n for c, n in null_counts.items() if n > 0}
    if any_nulls:
        print(f"\n  Null counts in numeric columns:")
        for c, n in any_nulls.items():
            print(f"    {c:<40s}: {n:>8,}  ({100*n/df.shape[0]:.1f}%)")
    else:
        print(f"\n  No nulls in any numeric column.")


def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    af2 = process_dataset(AF2_CSV, "af2")
    pdb = process_dataset(PDB_CSV, "pdb")

    print(f"\n{'─'*60}")
    print("Concatenating datasets ...")
    combined = pl.concat([af2, pdb], how="vertical")
    print(f"  Combined shape     : {combined.shape[0]:>8,} rows × {combined.shape[1]} cols")

    print(f"\nSaving to {OUT_PATH} ...")
    combined.write_parquet(OUT_PATH, compression="snappy")
    size_mb = OUT_PATH.stat().st_size / 1_048_576
    print(f"  Written: {size_mb:.1f} MB")

    print_summary(combined)
    print(f"\nDone. Output → {OUT_PATH}")


if __name__ == "__main__":
    main()
