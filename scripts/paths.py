"""
Rutas base del repositorio.

Ejecuta siempre desde la raíz del proyecto, por ejemplo:
  python scripts/benchmark_test1_assistments.py
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ASSISTMENTS = REPO_ROOT / "results" / "assistments"
FIGURAS_DIR = RESULTS_ASSISTMENTS / "figuras"
LOGS_DIR = RESULTS_ASSISTMENTS / "logs"
DATA_DIR = REPO_ROOT / "data"
SQL_DIR = REPO_ROOT / "sql"
DEFAULT_DATASET_CSV = DATA_DIR / "dataset.csv"
