"""
FastAPI serving layer for a3d-predictor.

Endpoints
---------
POST /predict          — single sequence
POST /predict_batch    — list of (id, sequence) pairs
GET  /health           — liveness / readiness check
POST /mutate           — submit a saturation-mutagenesis job (async)
GET  /mutate/{job_id}  — poll job status / retrieve results

Run with:
    uvicorn src.serving.api:app --host 0.0.0.0 --port 8000

Mutagenesis example (curl)
--------------------------
# Submit a job
curl -s -X POST http://localhost:8000/mutate \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKFLILLFNILCLFPVLAAD", "top_n": 5}' | python3 -m json.tool

# Poll until done (replace JOB_ID with the value from the previous response)
curl -s http://localhost:8000/mutate/JOB_ID | python3 -m json.tool

# Scan only positions 1, 5, and 10-15
curl -s -X POST http://localhost:8000/mutate \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKFLILLFNILCLFPVLAAD", "positions": [1, 5, 10, 11, 12, 13, 14, 15]}' \
  | python3 -m json.tool
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_VALID_AA = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
_MAX_SEQ_LEN      = 1022   # ESM-2 hard limit (1024 tokens − 2 BOS/EOS)
_MAX_BATCH        = 64     # guard against oversized batch requests
_MUTATE_BATCH     = 8      # ESM-2 batch size for mutagenesis forward passes
_MUTATE_MAX_JOBS  = 256    # cap in-memory job store to avoid unbounded growth

log = logging.getLogger("a3d_predictor.api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Shared app state (populated at startup)
# ---------------------------------------------------------------------------
_state: dict = {
    "predictor":     None,
    "model_version": "unknown",
    "model_loaded":  False,
    # Mutagenesis job store: job_id → job dict
    "jobs":          {},
    # Single-worker thread pool so GPU jobs don't race each other
    "executor":      None,
}


# ---------------------------------------------------------------------------
# Lifespan: load model once at startup, release on shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # One worker: mutagenesis jobs must not run concurrently on the same GPU.
    _state["executor"] = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mutate")

    log.info("Loading Predictor (ESM-2 + regression head) …")
    t0 = time.perf_counter()
    try:
        from src.serving.predict import Predictor
        predictor = Predictor()
        _state["predictor"]     = predictor
        _state["model_version"] = _build_model_version(predictor)
        _state["model_loaded"]  = True
        elapsed = time.perf_counter() - t0
        log.info("Predictor ready in %.1f s  [%s]", elapsed, _state["model_version"])
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to load predictor: %s", exc)
        # Server starts anyway; /health will report model_loaded=false
    yield
    log.info("Shutting down — releasing model.")
    _state["predictor"]    = None
    _state["model_loaded"] = False
    _state["executor"].shutdown(wait=False)
    _state["executor"] = None


def _build_model_version(predictor) -> str:
    """Derive a human-readable version tag from the loaded backend."""
    name: str = predictor._model_name   # e.g. "MLP (best_mlp.pt)"
    if name.startswith("MLP"):
        return "mlp_v1_esm2_650M"
    return "ridge_esm2_650M"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="A3D Predictor",
    description="Predict Aggrescan3D average value from protein sequences.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

def _validate_sequence(seq: str) -> str:
    seq = seq.strip().upper()
    if not seq:
        raise ValueError("sequence must not be empty")
    if len(seq) > _MAX_SEQ_LEN:
        raise ValueError(
            f"sequence length {len(seq)} exceeds the maximum of {_MAX_SEQ_LEN} "
            "amino acids supported by ESM-2"
        )
    if not _VALID_AA.match(seq):
        bad = sorted({c for c in seq if c not in "ACDEFGHIKLMNPQRSTVWY"})
        raise ValueError(
            f"sequence contains invalid character(s): {bad}. "
            "Only standard one-letter amino-acid codes (A-Z minus BJOUXZ) are accepted."
        )
    return seq


class SingleRequest(BaseModel):
    sequence: Annotated[str, Field(min_length=1, description="Protein amino-acid sequence")]

    @field_validator("sequence")
    @classmethod
    def check_sequence(cls, v: str) -> str:
        return _validate_sequence(v)


class SingleResponse(BaseModel):
    protein_id:       str   = Field(description="Echo of the input identifier")
    predicted_a3d_avg: float = Field(description="Predicted Aggrescan3D average value")
    sequence_length:  int   = Field(description="Number of amino acids in the sequence")
    model_version:    str   = Field(description="Model checkpoint identifier")


class SequenceItem(BaseModel):
    id:       str = Field(description="Caller-supplied identifier for this sequence")
    sequence: Annotated[str, Field(min_length=1)]

    @field_validator("sequence")
    @classmethod
    def check_sequence(cls, v: str) -> str:
        return _validate_sequence(v)


class BatchRequest(BaseModel):
    sequences: Annotated[
        list[SequenceItem],
        Field(min_length=1, max_length=_MAX_BATCH, description="List of sequences to predict"),
    ]


class PredictionItem(BaseModel):
    protein_id:        str
    predicted_a3d_avg: float
    sequence_length:   int
    model_version:     str


class BatchResponse(BaseModel):
    predictions: list[PredictionItem]


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    model_version: str


# -- Mutagenesis schemas -----------------------------------------------------

class MutateRequest(BaseModel):
    sequence: Annotated[str, Field(min_length=1, description="Wild-type protein sequence")]
    top_n: int = Field(
        default=20,
        ge=1,
        le=500,
        description="How many top mutations to include in the response",
    )
    positions: list[int] | None = Field(
        default=None,
        description=(
            "1-based positions to scan. "
            "Pass null / omit to scan every position in the sequence."
        ),
    )

    @field_validator("sequence")
    @classmethod
    def check_sequence(cls, v: str) -> str:
        return _validate_sequence(v)

    @field_validator("positions")
    @classmethod
    def check_positions(cls, v: list[int] | None) -> list[int] | None:
        if v is None:
            return v
        bad = [p for p in v if p < 1]
        if bad:
            raise ValueError(f"positions must be ≥ 1; got {bad[:5]}")
        return sorted(set(v))


class MutateJobResponse(BaseModel):
    """Returned immediately when a job is submitted."""
    job_id:            str
    status:            str   = "running"
    estimated_seconds: float = Field(description="Rough wall-clock estimate for the job")


class TopMutation(BaseModel):
    position:              int
    wildtype_aa:           str
    mutant_aa:             str
    predicted_a3d_mutant:  float
    delta_a3d:             float


class PositionSummaryItem(BaseModel):
    position:          int
    wildtype_aa:       str
    avg_delta:         float
    best_substitution: str


class MutateResult(BaseModel):
    wildtype_predicted_a3d: float
    sequence_length:        int
    num_mutations_scanned:  int
    top_mutations:          list[TopMutation]
    position_summary:       list[PositionSummaryItem]


class MutateStatusResponse(BaseModel):
    """Returned by GET /mutate/{job_id}."""
    job_id:            str
    status:            str         # "running" | "done" | "failed"
    submitted_at:      float       # Unix timestamp
    finished_at:       float | None = None
    estimated_seconds: float
    result:            MutateResult | None = None
    error:             str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness and readiness check",
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if _state["model_loaded"] else "degraded",
        model_loaded=_state["model_loaded"],
        model_version=_state["model_version"],
    )


@app.post(
    "/predict",
    response_model=SingleResponse,
    summary="Predict aggrescan3d_avg_value for a single sequence",
)
def predict(body: SingleRequest) -> SingleResponse:
    _require_model()
    try:
        value = _state["predictor"].predict(body.sequence)
    except Exception as exc:  # noqa: BLE001
        log.exception("Prediction error for single sequence")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc

    return SingleResponse(
        protein_id="user_input",
        predicted_a3d_avg=round(float(value), 6),
        sequence_length=len(body.sequence),
        model_version=_state["model_version"],
    )


@app.post(
    "/predict_batch",
    response_model=BatchResponse,
    summary="Predict aggrescan3d_avg_value for a batch of sequences",
)
def predict_batch(body: BatchRequest) -> BatchResponse:
    _require_model()
    ids   = [item.id       for item in body.sequences]
    seqs  = [item.sequence for item in body.sequences]
    try:
        preds = _state["predictor"].predict_batch(seqs, batch_size=8)
    except Exception as exc:  # noqa: BLE001
        log.exception("Prediction error for batch of %d sequences", len(seqs))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {exc}",
        ) from exc

    mv = _state["model_version"]
    return BatchResponse(
        predictions=[
            PredictionItem(
                protein_id=pid,
                predicted_a3d_avg=round(float(p), 6),
                sequence_length=len(seq),
                model_version=mv,
            )
            for pid, seq, p in zip(ids, seqs, preds)
        ]
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_model() -> None:
    if not _state["model_loaded"] or _state["predictor"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Check /health for details.",
        )


def _run_mutagenesis_job(
    job_id: str,
    sequence: str,
    positions_0based: list[int],
    top_n: int,
) -> None:
    """
    Blocking function executed in the thread pool.
    Updates _state["jobs"][job_id] in-place as it runs.
    """
    from src.serving.mutate import iter_mutants, per_position_summary

    job = _state["jobs"][job_id]
    job["started_at"] = time.time()
    log.info("Job %s started  (%d positions × 19 mutations)", job_id, len(positions_0based))

    try:
        predictor = _state["predictor"]

        # --- wild-type score (1 forward pass) --------------------------------
        wt_score = float(predictor.predict(sequence))
        log.info("Job %s  wt_score=%.4f", job_id, wt_score)

        # --- collect all mutant sequences ------------------------------------
        mut_meta: list[tuple[int, str, str]] = []   # (pos_1indexed, wt_aa, mut_aa)
        mut_seqs: list[str] = []
        for pos1, wt_aa, mut_aa, mut_seq in iter_mutants(sequence, positions_0based):
            mut_meta.append((pos1, wt_aa, mut_aa))
            mut_seqs.append(mut_seq)

        n_muts = len(mut_seqs)
        job["num_mutations"] = n_muts

        # --- batch-score mutants ---------------------------------------------
        all_preds: list[float] = []
        for i in range(0, n_muts, _MUTATE_BATCH):
            batch = mut_seqs[i : i + _MUTATE_BATCH]
            embs  = predictor._embed_batch(batch)
            preds = predictor._run_head(embs)
            all_preds.extend(float(p) for p in preds)

        # --- assemble rows ---------------------------------------------------
        rows: list[dict] = []
        for (pos1, wt_aa, mut_aa), mut_score in zip(mut_meta, all_preds):
            rows.append(
                {
                    "position":               pos1,
                    "wildtype_aa":            wt_aa,
                    "mutant_aa":              mut_aa,
                    "predicted_a3d_wildtype": round(wt_score, 6),
                    "predicted_a3d_mutant":   round(float(mut_score), 6),
                    "delta_a3d":              round(float(mut_score) - wt_score, 6),
                }
            )
        rows.sort(key=lambda r: r["delta_a3d"])

        # --- position summary ------------------------------------------------
        pos_summary_raw = per_position_summary(rows)

        # Build pos → best_AA in a single O(n) pass
        best_by_pos: dict[int, tuple[float, str]] = {}
        for r in rows:
            p = r["position"]
            if p not in best_by_pos or r["delta_a3d"] < best_by_pos[p][0]:
                best_by_pos[p] = (r["delta_a3d"], r["mutant_aa"])

        position_summary = [
            PositionSummaryItem(
                position=s["position"],
                wildtype_aa=s["wildtype_aa"],
                avg_delta=s["mean_delta_a3d"],
                best_substitution=best_by_pos[s["position"]][1],
            )
            for s in pos_summary_raw
        ]

        # --- top mutations ---------------------------------------------------
        top_mutations = [
            TopMutation(
                position=r["position"],
                wildtype_aa=r["wildtype_aa"],
                mutant_aa=r["mutant_aa"],
                predicted_a3d_mutant=r["predicted_a3d_mutant"],
                delta_a3d=r["delta_a3d"],
            )
            for r in rows[:top_n]
        ]

        job["result"] = MutateResult(
            wildtype_predicted_a3d=round(wt_score, 6),
            sequence_length=len(sequence),
            num_mutations_scanned=n_muts,
            top_mutations=top_mutations,
            position_summary=position_summary,
        )
        job["status"] = "done"
        log.info("Job %s done  (%d mutations scored)", job_id, n_muts)

    except Exception as exc:  # noqa: BLE001
        log.exception("Job %s failed", job_id)
        job["error"]  = str(exc)
        job["status"] = "failed"
    finally:
        job["finished_at"] = time.time()


# ---------------------------------------------------------------------------
# Mutagenesis endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/mutate",
    response_model=MutateJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a saturation-mutagenesis job (returns immediately)",
)
async def mutate_submit(body: MutateRequest) -> MutateJobResponse:
    """
    Enqueue a saturation-mutagenesis scan and return a **job_id** immediately.

    The scan evaluates all 19 amino-acid substitutions at every requested
    position (or all positions when `positions` is null).  Use
    **GET /mutate/{job_id}** to poll for completion and retrieve the ranked
    mutation list.
    """
    _require_model()

    seq     = body.sequence
    seq_len = len(seq)

    # Resolve and validate positions (convert to 0-based)
    if body.positions is not None:
        out_of_range = [p for p in body.positions if p > seq_len]
        if out_of_range:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Positions out of range for sequence of length {seq_len}: "
                    f"{out_of_range[:10]}"
                ),
            )
        positions_0based = [p - 1 for p in body.positions]
    else:
        positions_0based = list(range(seq_len))

    n_muts = len(positions_0based) * 19
    on_gpu = _state["predictor"].device.type == "cuda"

    from src.serving.mutate import estimate_seconds
    est_secs = estimate_seconds(n_muts, _MUTATE_BATCH, on_gpu=on_gpu)

    # Evict oldest jobs if we're at the cap
    jobs = _state["jobs"]
    if len(jobs) >= _MUTATE_MAX_JOBS:
        oldest = sorted(jobs, key=lambda jid: jobs[jid]["submitted_at"])
        for jid in oldest[: len(jobs) - _MUTATE_MAX_JOBS + 1]:
            del jobs[jid]

    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {
        "status":            "running",
        "submitted_at":      time.time(),
        "started_at":        None,
        "finished_at":       None,
        "estimated_seconds": est_secs,
        "num_mutations":     n_muts,
        "result":            None,
        "error":             None,
    }

    # Fire-and-forget: run in the single-worker thread pool so the GPU is
    # accessed from only one thread at a time.
    asyncio.create_task(
        asyncio.to_thread(
            _run_mutagenesis_job,
            job_id,
            seq,
            positions_0based,
            body.top_n,
        )
    )

    log.info(
        "Job %s submitted  seq_len=%d  positions=%d  est=%.0fs",
        job_id, seq_len, len(positions_0based), est_secs,
    )
    return MutateJobResponse(
        job_id=job_id,
        status="running",
        estimated_seconds=round(est_secs, 1),
    )


@app.get(
    "/mutate/{job_id}",
    response_model=MutateStatusResponse,
    summary="Poll a mutagenesis job for status and results",
)
def mutate_status(job_id: str) -> MutateStatusResponse:
    """
    Return the current status of a mutagenesis job.

    * **status = "running"** — job is still in progress; poll again later.
    * **status = "done"** — `result` contains the ranked mutations.
    * **status = "failed"** — `error` contains the exception message.
    """
    job = _state["jobs"].get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )

    return MutateStatusResponse(
        job_id=job_id,
        status=job["status"],
        submitted_at=job["submitted_at"],
        finished_at=job["finished_at"],
        estimated_seconds=job["estimated_seconds"],
        result=job["result"],
        error=job["error"],
    )
