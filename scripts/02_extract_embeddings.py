"""
02_extract_embeddings.py
------------------------
Extract ESM-2 (esm2_t33_650M_UR50D, 650M params) embeddings for protein sequences
from train/val/test splits. Saves mean-pooled 1280-dim vectors to HDF5.

Supports incremental / resume-safe writing: if the output file already exists
and contains partially-processed sequences, the script picks up from where it
left off.

Usage:
    conda run -n a3d python scripts/02_extract_embeddings.py
    conda run -n a3d python scripts/02_extract_embeddings.py --splits train
    conda run -n a3d python scripts/02_extract_embeddings.py --splits train val test
"""

import argparse
import logging
import time
from pathlib import Path

import esm
import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEQ_LEN = 1022       # ESM-2 hard limit (1024 tokens − 2 for BOS/EOS)
BATCH_SIZE = 8
EMB_DIM = 1280
REPR_LAYER = 33          # last layer of esm2_t33_650M_UR50D
FLUSH_EVERY = 50         # flush HDF5 every N batches (= 400 seqs by default)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(device: torch.device):
    log.info("Loading esm2_t33_650M_UR50D …")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.half()          # float16 to save VRAM
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    log.info("Model loaded (float16, %s)", device)
    return model, alphabet, batch_converter


# ---------------------------------------------------------------------------
# HDF5 helpers (incremental / resume-safe)
# ---------------------------------------------------------------------------

def _open_or_create_h5(path: Path) -> tuple[h5py.File, h5py.Dataset, h5py.Dataset]:
    """
    Open (or create) a resizable HDF5 file.
    Returns (file, embeddings_dataset, design_names_dataset).
    Caller is responsible for closing the file.
    """
    f = h5py.File(path, "a")
    if "embeddings" not in f:
        emb_ds = f.create_dataset(
            "embeddings",
            shape=(0, EMB_DIM),
            maxshape=(None, EMB_DIM),
            dtype="float32",
            chunks=(min(512, 1), EMB_DIM),
            compression="lzf",        # fast, lossless
        )
        name_ds = f.create_dataset(
            "design_names",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
    else:
        emb_ds = f["embeddings"]
        name_ds = f["design_names"]
    return f, emb_ds, name_ds


def _append_to_h5(
    emb_ds: h5py.Dataset,
    name_ds: h5py.Dataset,
    embeddings: np.ndarray,   # shape [B, 1280]
    names: list[str],
) -> None:
    """Extend both datasets by the current batch in-place."""
    n = len(names)
    cur = emb_ds.shape[0]
    emb_ds.resize(cur + n, axis=0)
    emb_ds[cur : cur + n] = embeddings
    name_ds.resize(cur + n, axis=0)
    name_ds[cur : cur + n] = names


def _already_done(path: Path) -> set[str]:
    """Return the set of design_names already saved in `path`."""
    if not path.exists():
        return set()
    with h5py.File(path, "r") as f:
        if "design_names" not in f:
            return set()
        return set(n.decode() if isinstance(n, bytes) else n for n in f["design_names"][:])


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def mean_pool(
    token_repr: torch.Tensor,   # [B, T, D]  (T includes BOS + residues + EOS/pad)
    seq_lengths: list[int],     # actual residue counts (no special tokens)
) -> np.ndarray:
    """Mean-pool over residue positions (skip BOS at 0 and EOS/padding)."""
    pooled = np.empty((len(seq_lengths), token_repr.shape[-1]), dtype=np.float32)
    for i, slen in enumerate(seq_lengths):
        # positions 1 … slen  (inclusive) are the residue tokens
        pooled[i] = token_repr[i, 1 : slen + 1].float().mean(dim=0).cpu().numpy()
    return pooled


def process_split(
    split: str,
    model,
    batch_converter,
    device: torch.device,
) -> None:
    t0 = time.perf_counter()

    parquet_path = SPLITS_DIR / f"{split}.parquet"
    out_path = OUT_DIR / f"embeddings_{split}.h5"

    log.info("=== Split: %s  (source: %s) ===", split, parquet_path)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    df = pd.read_parquet(parquet_path, columns=["design_name", "full_sequence", "seq_length"])
    total_rows = len(df)

    # ------------------------------------------------------------------
    # Resume: skip already-processed sequences
    # ------------------------------------------------------------------
    done_names = _already_done(out_path)
    if done_names:
        log.info("Resuming — %d sequences already in %s", len(done_names), out_path.name)
        df = df[~df["design_name"].isin(done_names)]

    # ------------------------------------------------------------------
    # Filter sequences exceeding ESM-2 limit
    # ------------------------------------------------------------------
    too_long_mask = df["seq_length"] > MAX_SEQ_LEN
    n_skipped = int(too_long_mask.sum())
    if n_skipped:
        log.warning(
            "Skipping %d / %d sequences with seq_length > %d (%.1f%%)",
            n_skipped, total_rows, MAX_SEQ_LEN,
            100 * n_skipped / total_rows,
        )
    df = df[~too_long_mask].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Sort by length → minimise padding within each batch
    # ------------------------------------------------------------------
    df = df.sort_values("seq_length").reset_index(drop=True)
    n_to_process = len(df)
    log.info("Sequences to process this run: %d", n_to_process)

    if n_to_process == 0:
        log.info("Nothing to do for split '%s'. Exiting early.", split)
        return

    # ------------------------------------------------------------------
    # Incremental HDF5 writing
    # ------------------------------------------------------------------
    h5_file, emb_ds, name_ds = _open_or_create_h5(out_path)

    try:
        # Accumulate a mini-buffer between flushes
        buf_embs: list[np.ndarray] = []
        buf_names: list[str] = []

        n_processed = 0
        batch_count = 0
        peak_mem_mb = 0.0

        with tqdm(total=n_to_process, desc=f"[{split}]", unit="seq", dynamic_ncols=True) as pbar:
            for batch_start in range(0, n_to_process, BATCH_SIZE):
                batch_df = df.iloc[batch_start : batch_start + BATCH_SIZE]
                data = list(zip(batch_df["design_name"], batch_df["full_sequence"]))
                seq_lengths = batch_df["seq_length"].tolist()

                # ESM-2 batch conversion
                _, _, batch_tokens = batch_converter(data)
                batch_tokens = batch_tokens.to(device)

                with torch.no_grad():
                    results = model(
                        batch_tokens,
                        repr_layers=[REPR_LAYER],
                        return_contacts=False,
                    )

                token_repr = results["representations"][REPR_LAYER]  # [B, T, D]
                pooled = mean_pool(token_repr, seq_lengths)           # [B, 1280] float32

                buf_embs.append(pooled)
                buf_names.extend(batch_df["design_name"].tolist())

                n_processed += len(data)
                batch_count += 1
                pbar.update(len(data))

                # --- Flush to HDF5 every FLUSH_EVERY batches ---
                if batch_count % FLUSH_EVERY == 0:
                    _append_to_h5(emb_ds, name_ds, np.vstack(buf_embs), buf_names)
                    h5_file.flush()
                    buf_embs.clear()
                    buf_names.clear()

                # --- VRAM management ---
                if device.type == "cuda":
                    mem_mb = torch.cuda.memory_allocated(device) / 1e6
                    peak_mem_mb = max(peak_mem_mb, mem_mb)
                    if mem_mb > 6_000:          # >6 GB: pre-emptively free cache
                        torch.cuda.empty_cache()

        # Flush remainder
        if buf_embs:
            _append_to_h5(emb_ds, name_ds, np.vstack(buf_embs), buf_names)
            h5_file.flush()

    finally:
        h5_file.close()

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t0
    if device.type == "cuda":
        peak_mem_mb = max(
            peak_mem_mb,
            torch.cuda.max_memory_allocated(device) / 1e6,
        )
        torch.cuda.reset_peak_memory_stats(device)

    total_saved = len(done_names) + n_processed
    log.info("─" * 60)
    log.info("Split           : %s", split)
    log.info("Total rows      : %d", total_rows)
    log.info("Processed (run) : %d", n_processed)
    log.info("Skipped (>1022) : %d  (%.1f%%)", n_skipped, 100 * n_skipped / total_rows)
    log.info("Saved in HDF5   : %d", total_saved)
    log.info("Time taken      : %.1f s  (%.1f seq/s)", elapsed, n_processed / max(elapsed, 1e-6))
    if device.type == "cuda":
        log.info("GPU peak VRAM   : %.0f MB", peak_mem_mb)
    log.info("Output          : %s", out_path)
    log.info("─" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract ESM-2 embeddings for a3d-predictor splits.")
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Which splits to process (default: all three).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Sequences per forward pass (default: {BATCH_SIZE}).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(device))

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    model, _alphabet, batch_converter = load_model(device)

    for split in args.splits:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        process_split(split, model, batch_converter, device)

    log.info("All done.")


if __name__ == "__main__":
    main()
