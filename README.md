# A3D-Predictor: Sequence-based Aggrescan3D Prediction

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Predict per-protein Aggrescan3D (A3D) aggregation scores directly from amino-acid sequence using ESM-2 embeddings and a trained regression head — no 3D structure required.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/your-username/a3d-predictor.git
cd a3d-predictor

# 2. Create and activate the conda environment
conda env create -f environment.yml
conda activate a3d

# 3. Install the package in editable mode
pip install -e .

# 4. Predict a single sequence
python -m src.serving.cli --sequence "MKFLILLFNILCLFPVLAADQSTDDNKQITTVNALLKYIAQSPAQSATQPQDAMTQNFSTQPKTSQNNAEPAVLSGNGDPNLSQRSQNAKESQASP"
```

---

## Installation

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- CUDA-capable GPU recommended (CPU inference works but is slower)

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/your-username/a3d-predictor.git
cd a3d-predictor
```

**2. Create the conda environment**

```bash
conda env create -f environment.yml
conda activate a3d
```

**3. Install in editable mode**

```bash
pip install -e .
```

**4. Model weights**

The Ridge regression checkpoint (`models_saved/ridge_esm.pkl`) and XGBoost model (`models_saved/xgb_esm.json`) are included in the repository (< 4 MB total). ESM-2 weights (~1.3 GB) are downloaded automatically on first run via `fair-esm` and cached by PyTorch Hub.

If a trained MLP checkpoint (`models_saved/best_mlp.pt`) is present, it takes priority over the Ridge model at inference time.

---

## Usage

### Single sequence (CLI)

```bash
python -m src.serving.cli --sequence "MKFLILLFNILCLFPVLAAD..."
# Output: predicted_a3d_avg: 0.342819
```

### FASTA file (CLI)

```bash
# Print to stdout
python -m src.serving.cli --fasta test_sequences.fasta

# Write to CSV
python -m src.serving.cli --fasta test_sequences.fasta --output predictions.csv
# Columns: protein_id, predicted_a3d_avg, sequence_length
```

### REST API server

```bash
# Start the server (default: http://0.0.0.0:8000)
./run_api.sh

# Or with custom host/port
./run_api.sh --host 127.0.0.1 --port 9000
```

Interactive API docs are available at `http://localhost:8000/docs`.

**Example requests:**

```bash
# Single prediction
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKFLILLFNILCLFPVLAAD"}' | python3 -m json.tool

# Batch prediction
curl -s -X POST http://localhost:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"sequences": [{"id": "prot1", "sequence": "MKFLIL..."}, {"id": "prot2", "sequence": "ACDEF..."}]}' \
  | python3 -m json.tool

# Health check
curl http://localhost:8000/health
```

### Mutation scanning

Identify single-point mutations that reduce predicted aggregation propensity:

```bash
# Full saturation scan
python -m src.serving.mutate --sequence "MKFLILLFNILCLFPVLAAD..." --output mutations.csv

# Scan specific positions (1-based, ranges supported)
python -m src.serving.mutate --sequence "MKFLILLFNILCLFPVLAAD..." \
    --positions 10,15,20-30 --output mutations.csv

# FASTA input — one CSV per protein
python -m src.serving.mutate --fasta proteins.fasta --output results/

# Via the API (async job)
curl -s -X POST http://localhost:8000/mutate \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKFLILLFNILCLFPVLAAD", "top_n": 10}' | python3 -m json.tool
# Then poll: curl http://localhost:8000/mutate/{job_id}
```

---

## Method

```
Amino-acid sequence
       │
       ▼
 ESM-2 (650M)                ← fair-esm  esm2_t33_650M_UR50D
 Layer-33 representations
       │  mean-pool over residue positions
       ▼
 1280-dim embedding (float32)
       │
       ▼
 Regression head             ← Ridge (R²=0.916) or MLP (R²=TBD)
       │
       ▼
 Predicted aggrescan3d_avg_value
```

**ESM-2** ([Lin et al., 2023](https://doi.org/10.1126/science.ade2574)) is a 650 M-parameter protein language model trained on UniRef50. The last transformer layer's token representations are mean-pooled over residue positions (BOS/EOS excluded) to produce a fixed-length sequence embedding.

The regression head is either:
- **Ridge** — `sklearn.linear_model.RidgeCV` fitted on scaled ESM-2 embeddings (included in repo).
- **MLP** — a 4-layer feed-forward network (1280 → 512 → 256 → 128 → 1) with BatchNorm, GELU activations, and dropout, trained with PyTorch.

Training data: DE-STRESS database ([Atkinson et al.](https://doi.org/10.1016/j.str.2022.12.011)) — ~50k proteins with A3D scores computed on AlphaFold2 and experimental PDB structures.

---

## Performance

Test-set metrics (held-out split, ~10% of DE-STRESS):

| Model | R² | RMSE | MAE | Pearson *r* | Spearman *ρ* |
|---|---|---|---|---|---|
| Ridge (ESM-2) | 0.916 | 0.152 | 0.106 | 0.957 | 0.934 |
| XGBoost (ESM-2) | 0.907 | 0.160 | 0.112 | 0.952 | 0.924 |

Evaluation plots are in [results/](results/).

---

## Reproducing results

```bash
# 1. Download raw data (see instructions)
python scripts/download_data.py

# 2. Extract ESM-2 embeddings (~3 GB, requires GPU)
python scripts/02_extract_embeddings.py

# 3. Train baseline models
python scripts/03b_train_baseline_esm.py

# 4. Evaluate
python scripts/05_evaluate.py
```

---

## Citation

If you use A3D-Predictor in your research, please cite:

```bibtex
@software{a3d_predictor_2026,
  author  = {A3D-Predictor Contributors},
  title   = {A3D-Predictor: Sequence-based Aggrescan3D Prediction},
  year    = {2026},
  url     = {https://github.com/your-username/a3d-predictor},
}
```

This work relies on:

- **ESM-2**: Lin et al. (2023). *Evolutionary-scale prediction of atomic-level protein structure with a language model.* Science. https://doi.org/10.1126/science.ade2574
- **Aggrescan3D**: Iglesias et al. (2019). *Aggrescan3D (A3D) 2.0: prediction and engineering of protein solubility.* Nucleic Acids Research. https://doi.org/10.1093/nar/gkz321
- **DE-STRESS**: Atkinson et al. (2023). *Using DE-STRESS to guide protein engineering.* Structure. https://doi.org/10.1016/j.str.2022.12.011

---

## License

MIT License. See [LICENSE](LICENSE) for details.
