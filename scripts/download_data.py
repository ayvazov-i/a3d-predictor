#!/usr/bin/env python3
"""
Print instructions for downloading the DE-STRESS dataset used to train A3D-Predictor.

The raw data files are NOT included in the repository because they are large
(~hundreds of MB).  Run this script to see where and how to obtain them.
"""

INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════════════════╗
║            A3D-Predictor — Dataset Download Instructions                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

The model was trained on the DE-STRESS (Descriptors for Evaluating the
STRucturES of proteins) database, maintained by the Wells Wood Research Group
at the University of Edinburgh.

──────────────────────────────────────────────────────────────────────────────
 Where to download
──────────────────────────────────────────────────────────────────────────────

  Website  : https://pragmaticproteindesign.bio.ed.ac.uk/de-stress/
  Direct   : https://pragmaticproteindesign.bio.ed.ac.uk/de-stress/dataset

  Two separate CSV dumps are used in this project:
    • destress_data_pdb_082024.csv   — PDB-structure entries  (August 2024)
    • destress_data_af2.csv          — AlphaFold2-structure entries

  Each row contains a protein chain (amino-acid sequence) together with
  various computed descriptors, including aggrescan3d_avg_value — the
  prediction target used here.

  The matching FASTA files can be generated with src/data/parse_destress.py
  or downloaded if provided on the DE-STRESS site.

──────────────────────────────────────────────────────────────────────────────
 Where to put the files
──────────────────────────────────────────────────────────────────────────────

  Place the downloaded files in:

    data/raw/destress_data_pdb_082024.csv
    data/raw/destress_data_af2.csv

  The data/raw/ directory is excluded from version control (.gitignore).

──────────────────────────────────────────────────────────────────────────────
 Next steps after downloading
──────────────────────────────────────────────────────────────────────────────

  1. Parse and combine the CSVs:
       python -c "from src.data.parse_destress import main; main()"

  2. Extract ESM-2 embeddings (~3 GB — GPU recommended):
       python scripts/02_extract_embeddings.py

  3. Train models:
       python scripts/03b_train_baseline_esm.py

  4. Evaluate:
       python scripts/05_evaluate.py

──────────────────────────────────────────────────────────────────────────────
 Citation for DE-STRESS
──────────────────────────────────────────────────────────────────────────────

  Atkinson et al. (2023). Using DE-STRESS to guide protein engineering.
  Structure, 31(4), 448–457.e3.
  https://doi.org/10.1016/j.str.2022.12.011

"""

if __name__ == "__main__":
    print(INSTRUCTIONS)
