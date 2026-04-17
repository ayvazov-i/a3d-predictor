"""
Serving predictor: load best model checkpoint + ESM-2 and predict
aggrescan3d_avg_value for one or more protein sequences.

Model priority (first found wins):
  1. MLP   — models_saved/best_mlp.pt      (if present)
  2. Ridge — models_saved/ridge_esm.pkl    (best evaluated: R²=0.916 on test)

ESM-2 details:
  Model : esm2_t33_650M_UR50D  (650M params, 33-layer transformer)
  Layer : 33 (last)
  Pool  : mean over residue positions 1..seq_len (skip BOS/EOS tokens)
  Dim   : 1280-dim float32 vector
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]   # ~/a3d-predictor
MODELS_DIR = ROOT / "models_saved"

# Add project root to sys.path so `src.models.mlp` is importable regardless of
# the working directory from which the module is invoked.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# ESM-2 constants (must match training in scripts/02_extract_embeddings.py)
# ---------------------------------------------------------------------------
_ESM_MODEL_NAME = "esm2_t33_650M_UR50D"
_ESM_REPR_LAYER = 33
_ESM_MAX_SEQ_LEN = 1022   # 1024 tokens − 2 special (BOS/EOS)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_esm(device: torch.device):
    """Load ESM-2 (float16) and return (model, batch_converter)."""
    import esm as _esm  # fair-esm package

    model, alphabet = _esm.pretrained.esm2_t33_650M_UR50D()
    model = model.half().to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter


def _mean_pool(token_repr: torch.Tensor, seq_lengths: list[int]) -> np.ndarray:
    """
    Mean-pool over residue positions for a batch of sequences.

    Args:
        token_repr  : [B, T, D] float tensor (from ESM-2 repr layer)
        seq_lengths : actual residue counts (excluding BOS/EOS)

    Returns:
        float32 ndarray of shape [B, D]
    """
    D = token_repr.shape[-1]
    pooled = np.empty((len(seq_lengths), D), dtype=np.float32)
    for i, slen in enumerate(seq_lengths):
        # positions 1…slen are residue tokens (position 0 is BOS)
        pooled[i] = token_repr[i, 1 : slen + 1].float().mean(dim=0).cpu().numpy()
    return pooled


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class Predictor:
    """
    Unified predictor: ESM-2 embedding → best available regression head.

    Usage::

        pred = Predictor()
        value = pred.predict("MKFLILLFNILCLFPVLAAD...")
        values = pred.predict_batch(["SEQ1", "SEQ2", ...], batch_size=8)
    """

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[predict] Loading ESM-2 ({_ESM_MODEL_NAME}) on {self.device} …", flush=True)
        self.esm_model, self.batch_converter = _load_esm(self.device)

        self._model_name, self._backend = self._load_best_model()
        print(f"[predict] Regression head : {self._model_name}", flush=True)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_best_model(self) -> tuple[str, object]:
        """Return (name, backend) for the best available checkpoint."""
        mlp_path = MODELS_DIR / "best_mlp.pt"
        if mlp_path.exists():
            return self._load_mlp(mlp_path)

        ridge_path = MODELS_DIR / "ridge_esm.pkl"
        if ridge_path.exists():
            return self._load_ridge(ridge_path)

        raise FileNotFoundError(
            f"No model checkpoint found in {MODELS_DIR}. "
            "Expected best_mlp.pt or ridge_esm.pkl."
        )

    def _load_mlp(self, path: Path) -> tuple[str, object]:
        from src.models.mlp import MLPRegressor

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = (
            ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        )
        cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        model = MLPRegressor(
            input_dim=cfg.get("input_dim", 1280),
            hidden_dims=cfg.get("hidden_dims", (512, 256, 128)),
            dropout=0.0,   # disabled at inference
        ).to(self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return "MLP (best_mlp.pt)", ("mlp", model)

    def _load_ridge(self, path: Path) -> tuple[str, object]:
        with open(path, "rb") as fh:
            bundle = pickle.load(fh)
        return "Ridge (ridge_esm.pkl)", ("ridge", bundle)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed_batch(self, sequences: list[str]) -> np.ndarray:
        """
        Compute ESM-2 embeddings for a list of sequences (one ESM-2 forward pass).

        Sequences longer than _ESM_MAX_SEQ_LEN are silently truncated.

        Returns float32 ndarray of shape [N, 1280].
        """
        truncated = [s[:_ESM_MAX_SEQ_LEN] for s in sequences]
        seq_lengths = [len(s) for s in truncated]
        data = [(str(i), s) for i, s in enumerate(truncated)]

        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.esm_model(
                batch_tokens,
                repr_layers=[_ESM_REPR_LAYER],
                return_contacts=False,
            )

        token_repr = results["representations"][_ESM_REPR_LAYER]  # [B, T, 1280]
        return _mean_pool(token_repr, seq_lengths)

    # ------------------------------------------------------------------
    # Regression
    # ------------------------------------------------------------------

    def _run_head(self, X: np.ndarray) -> np.ndarray:
        """Run the regression head on a float32 [N, 1280] array."""
        kind, model = self._backend
        if kind == "mlp":
            tensor = torch.from_numpy(X).to(self.device)
            with torch.no_grad():
                preds = model(tensor).cpu().numpy()
            return preds.astype(np.float32)
        # ridge
        X_scaled = model["scaler"].transform(X)
        return model["model"].predict(X_scaled).astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, sequence: str) -> float:
        """
        Predict aggrescan3d_avg_value for a single protein sequence.

        Args:
            sequence: amino-acid string (standard one-letter codes)

        Returns:
            Predicted aggrescan3d_avg_value as a Python float.
        """
        emb = self._embed_batch([sequence])     # [1, 1280]
        return float(self._run_head(emb)[0])

    def predict_batch(
        self,
        sequences: list[str],
        batch_size: int = 8,
    ) -> np.ndarray:
        """
        Predict aggrescan3d_avg_value for a list of sequences.

        Args:
            sequences  : list of amino-acid strings
            batch_size : number of sequences per ESM-2 forward pass (default 8)

        Returns:
            float32 ndarray of shape [N] with one prediction per sequence.
        """
        all_embs: list[np.ndarray] = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            all_embs.append(self._embed_batch(batch))

        X = np.vstack(all_embs)          # [N, 1280]
        return self._run_head(X)         # [N]
