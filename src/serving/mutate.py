"""
In silico saturation mutagenesis for the A3D aggregation predictor.

For every position in a query protein (or a user-specified subset) this
script substitutes all 19 alternative amino acids, predicts the A3D score
with the trained model, and ranks mutations by

    delta_a3d = predicted_a3d_mutant - predicted_a3d_wildtype

The most negative delta_a3d values correspond to the largest reductions in
predicted aggregation propensity.

Usage
-----
Single sequence, full scan:
  python -m src.serving.mutate --sequence "MKFLIL..." --output mutations.csv

Single sequence, specific positions:
  python -m src.serving.mutate --sequence "MKFLIL..." --positions 10,15,20-30 \\
      --output mutations.csv

FASTA file (one CSV per protein written into an output directory):
  python -m src.serving.mutate --fasta proteins.fasta --output results/ --top 20

Skip the long-protein confirmation prompt:
  python -m src.serving.mutate --sequence "VERY_LONG..." --yes --output out.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AMINO_ACIDS: str = "ACDEFGHIKLMNPQRSTVWY"   # 20 standard AAs, alphabetical
_LONG_SEQ_THRESHOLD: int = 500              # warn if len(sequence) > this

# Time-per-batch rough estimates (seconds) used for the pre-run ETA
_SEC_PER_BATCH_GPU: float = 1.5
_SEC_PER_BATCH_CPU: float = 15.0


# ---------------------------------------------------------------------------
# FASTA parser (mirrors cli.py for self-containment)
# ---------------------------------------------------------------------------

def parse_fasta(fasta_path: Path) -> list[tuple[str, str]]:
    """Return list of (protein_id, sequence) parsed from *fasta_path*."""
    entries: list[tuple[str, str]] = []
    current_id: str | None = None
    seq_parts: list[str] = []

    with open(fasta_path) as fh:
        for line in fh:
            line = line.rstrip()
            if not line or line.startswith(";"):
                continue
            if line.startswith(">"):
                if current_id is not None:
                    entries.append((current_id, "".join(seq_parts)))
                current_id = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line.upper())

    if current_id is not None:
        entries.append((current_id, "".join(seq_parts)))

    return entries


# ---------------------------------------------------------------------------
# Position parser
# ---------------------------------------------------------------------------

def parse_positions(spec: str, seq_len: int) -> list[int]:
    """
    Parse a position specification string into a sorted list of 0-based indices.

    The spec uses 1-based positions, comma-separated, with optional ranges::

        "10"         → [9]
        "10,15"      → [9, 14]
        "20-30"      → [19, 20, …, 29]
        "10,15,20-30" → [9, 14, 19, …, 29]

    Out-of-range positions (< 1 or > seq_len) are silently discarded.
    """
    positions: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            try:
                start, end = int(parts[0]), int(parts[1])
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"Invalid position range: '{token}'"
                )
            for p in range(start, end + 1):
                if 1 <= p <= seq_len:
                    positions.add(p - 1)   # convert to 0-based
        else:
            try:
                p = int(token)
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"Invalid position: '{token}'"
                )
            if 1 <= p <= seq_len:
                positions.add(p - 1)

    return sorted(positions)


# ---------------------------------------------------------------------------
# Mutant generator
# ---------------------------------------------------------------------------

def iter_mutants(
    sequence: str,
    positions: list[int],
) -> Iterator[tuple[int, str, str, str]]:
    """
    Yield ``(pos_1indexed, wt_aa, mut_aa, mutant_sequence)`` for every
    single-point substitution at each position in *positions*.

    Only the 19 amino acids that differ from the wild-type residue are tried.
    """
    seq_list = list(sequence)
    for pos0 in positions:
        wt_aa = sequence[pos0]
        for alt_aa in AMINO_ACIDS:
            if alt_aa == wt_aa:
                continue
            seq_list[pos0] = alt_aa
            yield (pos0 + 1, wt_aa, alt_aa, "".join(seq_list))
            seq_list[pos0] = wt_aa   # restore


# ---------------------------------------------------------------------------
# Time estimator
# ---------------------------------------------------------------------------

def estimate_seconds(n_mutations: int, batch_size: int, on_gpu: bool) -> float:
    n_batches = -(-n_mutations // batch_size)   # ceiling division
    sec_per = _SEC_PER_BATCH_GPU if on_gpu else _SEC_PER_BATCH_CPU
    return n_batches * sec_per


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Core mutagenesis runner
# ---------------------------------------------------------------------------

def run_mutagenesis(
    predictor,
    sequence: str,
    positions: list[int],
    batch_size: int = 8,
    show_progress: bool = True,
) -> list[dict]:
    """
    Run saturation mutagenesis on *sequence* at all *positions*.

    Returns a list of dicts (one per mutation) with keys:
      position, wildtype_aa, mutant_aa,
      predicted_a3d_wildtype, predicted_a3d_mutant, delta_a3d

    The list is sorted by delta_a3d ascending (most beneficial first).
    """
    try:
        from tqdm import tqdm
        _tqdm_available = True
    except ImportError:
        _tqdm_available = False

    # ------------------------------------------------------------------
    # 1. Predict wild-type score (single forward pass)
    # ------------------------------------------------------------------
    print("[mutate] Predicting wild-type A3D score …", flush=True)
    wt_score = float(predictor.predict(sequence))
    print(f"[mutate] Wild-type predicted A3D: {wt_score:.6f}", flush=True)

    # ------------------------------------------------------------------
    # 2. Collect all mutant sequences
    # ------------------------------------------------------------------
    mut_meta: list[tuple[int, str, str]] = []   # (pos_1indexed, wt_aa, mut_aa)
    mut_seqs: list[str] = []

    for pos1, wt_aa, mut_aa, mut_seq in iter_mutants(sequence, positions):
        mut_meta.append((pos1, wt_aa, mut_aa))
        mut_seqs.append(mut_seq)

    n_muts = len(mut_seqs)
    if n_muts == 0:
        return []

    # ------------------------------------------------------------------
    # 3. Batch-predict all mutants with a progress bar
    # ------------------------------------------------------------------
    print(
        f"[mutate] Scoring {n_muts} mutants "
        f"({len(positions)} positions × 19 substitutions) …",
        flush=True,
    )

    all_preds: list[float] = []
    n_batches = -(-n_muts // batch_size)

    if show_progress and _tqdm_available:
        bar = tqdm(total=n_batches, unit="batch", desc="Mutagenesis")
    else:
        bar = None

    t0 = time.perf_counter()
    for i in range(0, n_muts, batch_size):
        batch = mut_seqs[i : i + batch_size]
        embs = predictor._embed_batch(batch)
        preds = predictor._run_head(embs)
        all_preds.extend(float(p) for p in preds)
        if bar is not None:
            bar.update(1)
        elif show_progress:
            done = min(i + batch_size, n_muts)
            pct = 100 * done / n_muts
            elapsed = time.perf_counter() - t0
            eta = elapsed / done * (n_muts - done) if done < n_muts else 0
            print(
                f"\r[mutate] {done}/{n_muts} ({pct:.1f}%)  "
                f"elapsed {format_duration(elapsed)}  "
                f"ETA {format_duration(eta)}",
                end="",
                flush=True,
            )

    if bar is not None:
        bar.close()
    elif show_progress:
        print()   # newline after \r progress

    elapsed = time.perf_counter() - t0
    print(f"[mutate] Scoring done in {format_duration(elapsed)}.", flush=True)

    # ------------------------------------------------------------------
    # 4. Assemble results
    # ------------------------------------------------------------------
    rows: list[dict] = []
    for (pos1, wt_aa, mut_aa), mut_score in zip(mut_meta, all_preds):
        rows.append(
            {
                "position": pos1,
                "wildtype_aa": wt_aa,
                "mutant_aa": mut_aa,
                "predicted_a3d_wildtype": round(wt_score, 6),
                "predicted_a3d_mutant": round(mut_score, 6),
                "delta_a3d": round(mut_score - wt_score, 6),
            }
        )

    rows.sort(key=lambda r: r["delta_a3d"])
    return rows


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def per_position_summary(rows: list[dict]) -> list[dict]:
    """
    For each position compute the mean and min delta_a3d across all 19
    substitutions.  Sorted by mean_delta_a3d ascending (most mutable first).
    """
    from collections import defaultdict

    pos_deltas: dict[int, list[float]] = defaultdict(list)
    pos_wt: dict[int, str] = {}
    for r in rows:
        pos_deltas[r["position"]].append(r["delta_a3d"])
        pos_wt[r["position"]] = r["wildtype_aa"]

    summary = []
    for pos, deltas in pos_deltas.items():
        arr = np.array(deltas)
        summary.append(
            {
                "position": pos,
                "wildtype_aa": pos_wt[pos],
                "mean_delta_a3d": round(float(arr.mean()), 6),
                "min_delta_a3d": round(float(arr.min()), 6),
                "max_delta_a3d": round(float(arr.max()), 6),
                "n_beneficial": int((arr < 0).sum()),
            }
        )

    summary.sort(key=lambda x: x["mean_delta_a3d"])
    return summary


def per_aa_summary(rows: list[dict]) -> list[dict]:
    """
    For each substituted amino acid compute the mean delta_a3d across all
    positions where it was tried.  Sorted ascending (most generally helpful AAs
    first).
    """
    from collections import defaultdict

    aa_deltas: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        aa_deltas[r["mutant_aa"]].append(r["delta_a3d"])

    summary = []
    for aa, deltas in aa_deltas.items():
        arr = np.array(deltas)
        summary.append(
            {
                "mutant_aa": aa,
                "mean_delta_a3d": round(float(arr.mean()), 6),
                "median_delta_a3d": round(float(np.median(arr)), 6),
                "min_delta_a3d": round(float(arr.min()), 6),
                "n_positions_tested": len(deltas),
                "n_beneficial": int((arr < 0).sum()),
            }
        )

    summary.sort(key=lambda x: x["mean_delta_a3d"])
    return summary


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_top_mutations(rows: list[dict], top_n: int, protein_id: str = "") -> None:
    header = f"  Top {top_n} mutations"
    if protein_id:
        header += f" for {protein_id}"
    print(header)
    print("  " + "-" * 68)
    print(
        f"  {'Rank':>4}  {'Pos':>5}  {'WT':>2} → {'Mut':>3}  "
        f"{'WT A3D':>10}  {'Mut A3D':>10}  {'ΔA3D':>10}"
    )
    print("  " + "-" * 68)
    for rank, r in enumerate(rows[:top_n], 1):
        print(
            f"  {rank:>4}  {r['position']:>5}  {r['wildtype_aa']:>2} → "
            f"{r['mutant_aa']:>3}  "
            f"{r['predicted_a3d_wildtype']:>10.4f}  "
            f"{r['predicted_a3d_mutant']:>10.4f}  "
            f"{r['delta_a3d']:>+10.4f}"
        )
    print()


def print_position_summary(summary: list[dict], top_n: int = 15) -> None:
    print(f"  Top {top_n} most mutable positions (by mean ΔA3D across all substitutions)")
    print("  " + "-" * 62)
    print(
        f"  {'Pos':>5}  {'WT':>2}  {'Mean ΔA3D':>10}  "
        f"{'Min ΔA3D':>10}  {'# Beneficial':>13}"
    )
    print("  " + "-" * 62)
    for row in summary[:top_n]:
        print(
            f"  {row['position']:>5}  {row['wildtype_aa']:>2}  "
            f"{row['mean_delta_a3d']:>+10.4f}  "
            f"{row['min_delta_a3d']:>+10.4f}  "
            f"{row['n_beneficial']:>13}"
        )
    print()


def print_aa_summary(summary: list[dict]) -> None:
    print("  Substitution tendency (mean ΔA3D per mutant AA across all positions)")
    print("  " + "-" * 62)
    print(
        f"  {'AA':>4}  {'Mean ΔA3D':>10}  {'Median ΔA3D':>12}  "
        f"{'# Positions':>11}  {'# Beneficial':>13}"
    )
    print("  " + "-" * 62)
    for row in summary:
        print(
            f"  {row['mutant_aa']:>4}  "
            f"{row['mean_delta_a3d']:>+10.4f}  "
            f"{row['median_delta_a3d']:>+12.4f}  "
            f"{row['n_positions_tested']:>11}  "
            f"{row['n_beneficial']:>13}"
        )
    print()


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

_MUT_FIELDS = [
    "position", "wildtype_aa", "mutant_aa",
    "predicted_a3d_wildtype", "predicted_a3d_mutant", "delta_a3d",
]

_POS_FIELDS = [
    "position", "wildtype_aa",
    "mean_delta_a3d", "min_delta_a3d", "max_delta_a3d", "n_beneficial",
]

_AA_FIELDS = [
    "mutant_aa", "mean_delta_a3d", "median_delta_a3d",
    "min_delta_a3d", "n_positions_tested", "n_beneficial",
]


def write_mutation_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_MUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Long-sequence confirmation
# ---------------------------------------------------------------------------

def _confirm_long_sequence(seq_len: int, n_muts: int, batch_size: int, on_gpu: bool) -> bool:
    """Ask the user if they want to proceed for long sequences."""
    eta = estimate_seconds(n_muts, batch_size, on_gpu)
    print(
        f"\n  WARNING: sequence length {seq_len} > {_LONG_SEQ_THRESHOLD}.\n"
        f"  This will evaluate {n_muts} mutants "
        f"(estimated {format_duration(eta)}).\n",
        flush=True,
    )
    try:
        answer = input("  Proceed? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""
    return answer in {"y", "yes"}


# ---------------------------------------------------------------------------
# Single-protein workflow
# ---------------------------------------------------------------------------

def process_one(
    predictor,
    protein_id: str,
    sequence: str,
    positions_spec: str | None,
    batch_size: int,
    top_n: int,
    output_path: Path | None,
    yes: bool,
    on_gpu: bool,
) -> list[dict] | None:
    """
    Run mutagenesis for a single protein.

    Returns the sorted mutation rows (or None if the user declined to proceed).
    """
    sequence = sequence.upper().strip()
    seq_len = len(sequence)

    # Resolve positions
    if positions_spec:
        positions = parse_positions(positions_spec, seq_len)
        if not positions:
            print(
                f"[mutate] WARNING: no valid positions after parsing '{positions_spec}' "
                f"for sequence of length {seq_len}. Skipping {protein_id}.",
                file=sys.stderr,
            )
            return None
    else:
        positions = list(range(seq_len))

    n_muts = len(positions) * 19

    # Estimate run time and print before starting
    eta = estimate_seconds(n_muts, batch_size, on_gpu)
    device_label = "GPU" if on_gpu else "CPU"
    print(
        f"\n[mutate] Protein     : {protein_id}\n"
        f"[mutate] Length      : {seq_len} residues\n"
        f"[mutate] Positions   : {len(positions)}\n"
        f"[mutate] Mutants     : {n_muts}  "
        f"({len(positions)} × 19)\n"
        f"[mutate] Batches     : {-(-n_muts // batch_size)}  "
        f"(batch_size={batch_size})\n"
        f"[mutate] Device      : {device_label}\n"
        f"[mutate] Estimated   : ~{format_duration(eta)} "
        f"({_SEC_PER_BATCH_GPU if on_gpu else _SEC_PER_BATCH_CPU:.1f} s/batch estimate)\n",
        flush=True,
    )

    # Long-protein guard
    if seq_len > _LONG_SEQ_THRESHOLD and not yes:
        if not _confirm_long_sequence(seq_len, n_muts, batch_size, on_gpu):
            print(f"[mutate] Skipped {protein_id}.", flush=True)
            return None

    # Run
    rows = run_mutagenesis(
        predictor, sequence, positions, batch_size=batch_size, show_progress=True
    )

    if not rows:
        print(f"[mutate] No mutations generated for {protein_id}.", flush=True)
        return rows

    # ---- Console output ---------------------------------------------------
    print()
    print_top_mutations(rows, top_n=top_n, protein_id=protein_id)

    pos_summary = per_position_summary(rows)
    print_position_summary(pos_summary, top_n=min(15, len(pos_summary)))

    aa_summary = per_aa_summary(rows)
    print_aa_summary(aa_summary)

    # ---- File output -------------------------------------------------------
    if output_path is not None:
        write_mutation_csv(rows, output_path)
        print(
            f"[mutate] Full results ({len(rows)} mutations) saved to {output_path}",
            flush=True,
        )

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.serving.mutate",
        description=(
            "In silico saturation mutagenesis: predict which single-point "
            "mutations reduce the A3D aggregation score most."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full scan of a single sequence
  python -m src.serving.mutate --sequence "MKFLILLFNILCLFPVLAAD..." \\
      --output mutations.csv

  # Scan only specific positions (1-indexed, ranges supported)
  python -m src.serving.mutate --sequence "MKFLILLFNILCLFPVLAAD..." \\
      --positions 10,15,20-30 --output mutations.csv

  # FASTA input — one CSV per protein written into results/
  python -m src.serving.mutate --fasta proteins.fasta --output results/ --top 20

  # Skip the long-sequence confirmation prompt
  python -m src.serving.mutate --sequence "VERY_LONG..." --yes --output out.csv
""",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--sequence",
        metavar="SEQ",
        type=str,
        help="Single protein sequence (one-letter amino-acid codes).",
    )
    mode.add_argument(
        "--fasta",
        metavar="FILE",
        type=Path,
        help=(
            "FASTA file with one or more sequences.  "
            "--output must be a directory when using this mode."
        ),
    )

    parser.add_argument(
        "--output",
        metavar="PATH",
        type=Path,
        default=None,
        help=(
            "Output path.  "
            "For --sequence mode: path to a CSV file.  "
            "For --fasta mode: path to an output directory (one CSV per protein)."
        ),
    )
    parser.add_argument(
        "--top",
        metavar="N",
        type=int,
        default=20,
        help="Number of top mutations to display (default: 20).",
    )
    parser.add_argument(
        "--batch-size",
        metavar="N",
        type=int,
        default=8,
        help="ESM-2 forward-pass batch size (default: 8).",
    )
    parser.add_argument(
        "--positions",
        metavar="SPEC",
        type=str,
        default=None,
        help=(
            "Comma-separated list of 1-based positions (ranges allowed, e.g. "
            "'10,15,20-30').  "
            "Omit to scan all positions."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help=(
            f"Skip the confirmation prompt for sequences longer than "
            f"{_LONG_SEQ_THRESHOLD} residues."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load predictor (ESM-2 + regression head)
    # ------------------------------------------------------------------
    from src.serving.predict import Predictor
    import torch

    predictor = Predictor()
    on_gpu = predictor.device.type == "cuda"

    # ------------------------------------------------------------------
    # Single-sequence mode
    # ------------------------------------------------------------------
    if args.sequence is not None:
        protein_id = "user_input"
        output_path: Path | None = args.output

        process_one(
            predictor=predictor,
            protein_id=protein_id,
            sequence=args.sequence,
            positions_spec=args.positions,
            batch_size=args.batch_size,
            top_n=args.top,
            output_path=output_path,
            yes=args.yes,
            on_gpu=on_gpu,
        )
        return

    # ------------------------------------------------------------------
    # FASTA mode
    # ------------------------------------------------------------------
    fasta_path: Path = args.fasta
    if not fasta_path.exists():
        print(f"error: FASTA file not found: {fasta_path}", file=sys.stderr)
        sys.exit(1)

    entries = parse_fasta(fasta_path)
    if not entries:
        print(f"error: no sequences found in {fasta_path}", file=sys.stderr)
        sys.exit(1)

    print(
        f"[mutate] Loaded {len(entries)} sequence(s) from {fasta_path.name}",
        flush=True,
    )

    # Output must be a directory when processing multiple sequences
    out_dir: Path | None = args.output
    if out_dir is not None:
        if out_dir.exists() and out_dir.is_file():
            print(
                f"error: --output '{out_dir}' is a file, but FASTA mode "
                "requires a directory.",
                file=sys.stderr,
            )
            sys.exit(1)
        out_dir.mkdir(parents=True, exist_ok=True)

    for protein_id, sequence in entries:
        # Sanitise protein_id for use as a filename
        safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in protein_id)
        csv_path = (out_dir / f"{safe_id}_mutations.csv") if out_dir else None

        process_one(
            predictor=predictor,
            protein_id=protein_id,
            sequence=sequence,
            positions_spec=args.positions,
            batch_size=args.batch_size,
            top_n=args.top,
            output_path=csv_path,
            yes=args.yes,
            on_gpu=on_gpu,
        )
        print("-" * 72, flush=True)


if __name__ == "__main__":
    main()
