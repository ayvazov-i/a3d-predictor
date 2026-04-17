"""
sequence_features.py
--------------------
Hand-crafted amino-acid-level features derived from a protein sequence string.

Feature vector layout (27 features total):
  [0:20]  amino acid composition (fraction) for each standard AA, alphabetical
  [20]    sequence length
  [21]    mean Kyte-Doolittle hydrophobicity
  [22]    fraction charged residues  (D, E, K, R)
  [23]    fraction aromatic residues (F, W, Y)
  [24]    fraction tiny residues     (A, G, S)
  [25]    fraction small residues    (A, C, D, G, N, P, S, T, V)
  [26]    fraction disordered-prone  (placeholder: same as fraction tiny — swap
                                      for a proper scale if desired)
"""

from __future__ import annotations
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

STANDARD_AAS: list[str] = list("ACDEFGHIKLMNPQRSTVWY")  # 20 aa, alphabetical

# Kyte-Doolittle hydrophobicity scale (Kyte & Doolittle, 1982)
KD_SCALE: dict[str, float] = {
    "A":  1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C":  2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I":  4.5,
    "L":  3.8, "K": -3.9, "M":  1.9, "F":  2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V":  4.2,
}

CHARGED   = frozenset("DEKR")
AROMATIC  = frozenset("FWY")
TINY      = frozenset("AGS")
SMALL     = frozenset("ACDGNPSTV")

_AA_INDEX: dict[str, int] = {aa: i for i, aa in enumerate(STANDARD_AAS)}

# Pre-build KD lookup as a fixed-size array for fast vectorised access
_KD_ARR = np.array([KD_SCALE.get(aa, 0.0) for aa in STANDARD_AAS], dtype=np.float32)

N_FEATURES: int = 27
FEATURE_NAMES: list[str] = (
    [f"comp_{aa}" for aa in STANDARD_AAS]
    + [
        "seq_length",
        "mean_kd_hydrophobicity",
        "frac_charged",
        "frac_aromatic",
        "frac_tiny",
        "frac_small",
        "frac_disordered_prone",
    ]
)


# ── Single-sequence feature extraction ───────────────────────────────────────

def sequence_to_features(seq: str) -> np.ndarray:
    """
    Compute a 27-dimensional feature vector for a single protein sequence.

    Parameters
    ----------
    seq : str
        Protein sequence using single-letter amino-acid codes.
        Non-standard characters are silently ignored in composition /
        fraction features; they do not contribute to counts.

    Returns
    -------
    np.ndarray, shape (27,), dtype float32
    """
    seq = seq.upper()
    n_total = len(seq)

    # Count occurrences of each standard AA
    counts = np.zeros(20, dtype=np.float32)
    n_charged  = 0
    n_aromatic = 0
    n_tiny     = 0
    n_small    = 0
    kd_sum     = 0.0
    n_standard = 0

    for aa in seq:
        idx = _AA_INDEX.get(aa)
        if idx is not None:
            counts[idx] += 1
            n_standard  += 1
            kd_sum      += KD_SCALE[aa]
            if aa in CHARGED:   n_charged  += 1
            if aa in AROMATIC:  n_aromatic += 1
            if aa in TINY:      n_tiny     += 1
            if aa in SMALL:     n_small    += 1

    denom = n_standard if n_standard > 0 else 1

    composition = counts / denom
    mean_kd     = kd_sum / denom
    frac_ch     = n_charged  / denom
    frac_ar     = n_aromatic / denom
    frac_ti     = n_tiny     / denom
    frac_sm     = n_small    / denom

    feats = np.empty(N_FEATURES, dtype=np.float32)
    feats[:20]  = composition
    feats[20]   = float(n_total)
    feats[21]   = mean_kd
    feats[22]   = frac_ch
    feats[23]   = frac_ar
    feats[24]   = frac_ti
    feats[25]   = frac_sm
    feats[26]   = frac_ti      # disordered-prone placeholder (= tiny fraction)

    return feats


# ── Batch extraction ──────────────────────────────────────────────────────────

def sequences_to_features(
    sequences: list[str],
    show_progress: bool = False,
) -> np.ndarray:
    """
    Compute features for a list of protein sequences.

    Parameters
    ----------
    sequences : list[str]
    show_progress : bool
        If True, display a tqdm progress bar.

    Returns
    -------
    np.ndarray, shape (len(sequences), 27), dtype float32
    """
    n = len(sequences)
    out = np.empty((n, N_FEATURES), dtype=np.float32)

    iterable = sequences
    if show_progress:
        from tqdm import tqdm
        iterable = tqdm(sequences, desc="Computing features", unit="seq", ncols=80)

    for i, seq in enumerate(iterable):
        out[i] = sequence_to_features(seq)

    return out
