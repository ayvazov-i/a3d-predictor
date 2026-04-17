# A3D-Predictor convenience targets
# Run from the project root with the 'a3d' conda environment active.

.PHONY: install predict serve test help

# Default target
help:
	@echo ""
	@echo "  A3D-Predictor — available targets"
	@echo "  ─────────────────────────────────────────────────────────────"
	@echo "  make install          Create the conda env and install the package"
	@echo "  make predict SEQ=...  Predict A3D score for a single sequence"
	@echo "  make serve            Start the FastAPI server on port 8000"
	@echo "  make test             Run the test suite with pytest"
	@echo ""
	@echo "  Example:"
	@echo '    make predict SEQ="MKFLILLFNILCLFPVLAAD"'
	@echo ""

# ── Environment setup ────────────────────────────────────────────────────────

install:
	conda env create -f environment.yml || conda env update -f environment.yml
	conda run -n a3d pip install -e .
	@echo ""
	@echo "  Done.  Activate with:  conda activate a3d"

# ── Inference ────────────────────────────────────────────────────────────────

# Usage: make predict SEQ="MKFLILLFNILCLFPVLAAD..."
predict:
ifndef SEQ
	$(error SEQ is not set. Usage: make predict SEQ="MKFLIL...")
endif
	python -m src.serving.cli --sequence "$(SEQ)"

# ── Serving ──────────────────────────────────────────────────────────────────

serve:
	./run_api.sh

# ── Testing ──────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v
