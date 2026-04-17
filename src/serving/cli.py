"""
CLI for a3d-predictor serving.

Usage:
  Single sequence:
    python -m src.serving.cli --sequence "MKFLILLFNILCLFPVLAAD..."

  FASTA file:
    python -m src.serving.cli --fasta input.fasta --output predictions.csv

FASTA output CSV columns: protein_id, predicted_a3d_avg, sequence_length
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# FASTA parser
# ---------------------------------------------------------------------------

def parse_fasta(fasta_path: Path) -> list[tuple[str, str]]:
    """
    Parse a FASTA file and return a list of (protein_id, sequence) tuples.

    The protein_id is the first whitespace-delimited token on the header line
    (after the leading '>').  Blank / comment lines between records are ignored.
    """
    entries: list[tuple[str, str]] = []
    current_id: str | None = None
    seq_parts: list[str] = []

    with open(fasta_path) as fh:
        for line in fh:
            line = line.rstrip()
            if not line or line.startswith(";"):   # skip blank / comment lines
                continue
            if line.startswith(">"):
                if current_id is not None:
                    entries.append((current_id, "".join(seq_parts)))
                current_id = line[1:].split()[0]   # first token after '>'
                seq_parts = []
            else:
                seq_parts.append(line)

    if current_id is not None:
        entries.append((current_id, "".join(seq_parts)))

    return entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.serving.cli",
        description="Predict aggrescan3d_avg_value from protein sequence(s).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.serving.cli --sequence "MKFLILLFNILCLFPVLAAD..."
  python -m src.serving.cli --fasta input.fasta
  python -m src.serving.cli --fasta input.fasta --output predictions.csv
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
        help="Path to a FASTA file containing one or more sequences.",
    )

    parser.add_argument(
        "--output",
        metavar="FILE",
        type=Path,
        default=None,
        help=(
            "Output CSV path for FASTA mode "
            "(default: write to stdout). "
            "Columns: protein_id, predicted_a3d_avg, sequence_length."
        ),
    )
    parser.add_argument(
        "--batch-size",
        metavar="N",
        type=int,
        default=8,
        help="ESM-2 batch size for FASTA mode (default: 8).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Lazy import: only load models when we actually need them
    from src.serving.predict import Predictor
    predictor = Predictor()

    # ------------------------------------------------------------------
    # Single-sequence mode
    # ------------------------------------------------------------------
    if args.sequence is not None:
        pred = predictor.predict(args.sequence)
        print(f"predicted_a3d_avg: {pred:.6f}")
        return

    # ------------------------------------------------------------------
    # FASTA mode
    # ------------------------------------------------------------------
    if not args.fasta.exists():
        print(f"error: FASTA file not found: {args.fasta}", file=sys.stderr)
        sys.exit(1)

    entries = parse_fasta(args.fasta)
    if not entries:
        print(f"error: no sequences found in {args.fasta}", file=sys.stderr)
        sys.exit(1)

    print(f"[cli] Loaded {len(entries)} sequence(s) from {args.fasta.name}", flush=True)

    ids = [e[0] for e in entries]
    seqs = [e[1] for e in entries]

    preds = predictor.predict_batch(seqs, batch_size=args.batch_size)

    fieldnames = ["protein_id", "predicted_a3d_avg", "sequence_length"]
    rows = [
        {
            "protein_id": pid,
            "predicted_a3d_avg": f"{float(p):.6f}",
            "sequence_length": len(seq),
        }
        for pid, seq, p in zip(ids, seqs, preds)
    ]

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[cli] Predictions written to {args.output}  ({len(rows)} sequences)")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
