"""
Carga el Wine Quality dataset en PostgreSQL y genera datasets sintéticos
para replicar los experimentos del artículo:

  Vallejo-Cabrera et al., Rev. Fac. Ing., Vol. 34, No. 74 (2025)
  DOI: 10.19053/01211129.v34.n74.2025.20737

Dataset original:
  https://www.kaggle.com/datasets/taweilo/wine-quality-dataset-balanced-classification
  (21 000 registros, 12 columnas: 11 fisicoquímicas + quality)

Uso:
  1. Descarga el CSV de Kaggle y colócalo en la ruta indicada en WINE_CSV_PATH.
  2. Ejecuta: python cargar_wine_quality.py
"""

import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# =====================================================
# CONFIGURACIÓN
# =====================================================
DB_USER     = "postgres"
DB_PASSWORD = "danonino32"
DB_HOST     = "127.0.0.1"
DB_PORT     = "5432"
DB_NAME     = "wine_quality"

# Pon aquí la ruta al CSV descargado de Kaggle
WINE_CSV_PATH = r"C:\Users\deat_\Downloads\fuentescartelcecen\dataset_ASSISTment\wine_quality.csv"

# Columnas fisicoquímicas (sin 'quality', que es la etiqueta)
FEATURE_COLS = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide",
    "density", "ph", "sulphates", "alcohol"
]

# Tamaños sintéticos del Test 1 del artículo
SYNTHETIC_SIZES = [1_000, 2_000, 5_000, 10_000, 21_000, 50_000, 100_000, 500_000, 1_000_000]

# =====================================================
# CONEXIÓN
# =====================================================
def get_engine(db_name=DB_NAME):
    return create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{db_name}",
        pool_pre_ping=True,
    )


def create_database_if_not_exists():
    """Crea la base de datos wine_quality si no existe."""
    engine = get_engine("postgres")
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        result = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :db"), {"db": DB_NAME}
        )
        if not result.fetchone():
            conn.execute(text(f'CREATE DATABASE "{DB_NAME}"'))
            print(f"[+] Base de datos '{DB_NAME}' creada.")
        else:
            print(f"[~] Base de datos '{DB_NAME}' ya existe.")


# =====================================================
# CARGA DEL DATASET BASE
# =====================================================
def load_base_dataset():
    if not os.path.exists(WINE_CSV_PATH):
        print(f"\n[!] ARCHIVO NO ENCONTRADO: {WINE_CSV_PATH}")
        print("    Descarga el dataset desde:")
        print("    https://www.kaggle.com/datasets/taweilo/wine-quality-dataset-balanced-classification")
        print("    y colócalo en la ruta indicada en WINE_CSV_PATH.\n")
        return None

    df = pd.read_csv(WINE_CSV_PATH)

    # Normalizar nombres de columnas (el CSV de Kaggle puede usar espacios o mayúsculas)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.dropna()

    # Verificar que existen las columnas esperadas
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"[!] Columnas no encontradas en el CSV: {missing}")
        print(f"    Columnas disponibles: {list(df.columns)}")
        return None

    print(f"[✓] Dataset cargado: {len(df):,} filas, columnas: {list(df.columns)}")
    return df


# =====================================================
# GENERACIÓN DE DATASETS SINTÉTICOS
# =====================================================
def generate_synthetic(df_base: pd.DataFrame, size: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Genera un dataset sintético de 'size' registros muestreando con reemplazo
    del dataset base y añadiendo ruido gaussiano pequeño para variabilidad.
    Mantiene las distribuciones originales de las 11 variables.
    """
    if size <= len(df_base):
        seed = int(rng.integers(0, 10**9))
        return pd.DataFrame(
            df_base[FEATURE_COLS].sample(n=size, random_state=seed).values,
            columns=FEATURE_COLS,
        )

    # Para tamaños mayores al dataset base: resampleo con ruido
    n_repeats = size // len(df_base) + 1
    df_big = pd.concat([df_base[FEATURE_COLS]] * n_repeats, ignore_index=True).iloc[:size].copy()

    # Ruido proporcional al 1% de la desviación estándar de cada columna
    noise_scale: np.ndarray = np.asarray(df_base[FEATURE_COLS].std()) * 0.01
    noise = rng.normal(0, noise_scale, size=df_big.shape)
    df_big = pd.DataFrame(df_big.to_numpy() + noise, columns=FEATURE_COLS)

    # Clip para mantener valores no negativos donde aplica
    for col in FEATURE_COLS:
        df_big[col] = df_big[col].clip(lower=0)

    return df_big.reset_index(drop=True)


# =====================================================
# CARGA EN POSTGRESQL
# =====================================================
def upload_datasets(df_base: pd.DataFrame):
    engine = get_engine()
    rng = np.random.default_rng(42)

    print("\n[+] Creando schema 'clustering' si no existe...")
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS clustering"))
        conn.commit()

    # Tabla base (21,000 registros originales = dataset de validación del artículo)
    print(f"\n[+] Subiendo tabla 'wine_quality_base' ({len(df_base):,} filas)...")
    df_base[FEATURE_COLS].to_sql(
        "wine_quality_base", engine,
        if_exists="replace", index=False, method="multi"
    )
    print(f"[✓] Tabla 'wine_quality_base' lista.")

    # Datasets sintéticos para Test 1
    for size in SYNTHETIC_SIZES:
        table_name = f"wine_synth_{size}"
        print(f"\n[+] Generando y subiendo '{table_name}' ({size:,} filas)...")
        df_synth = generate_synthetic(df_base, size, rng)
        df_synth.to_sql(
            table_name, engine,
            if_exists="replace", index=False, method="multi", chunksize=10_000
        )
        print(f"[✓] Tabla '{table_name}' lista: {len(df_synth):,} filas.")

    print("\n[✓] Todos los datasets cargados en PostgreSQL.")
    print(f"    Base de datos: {DB_NAME}")
    print(f"    Tablas creadas: wine_quality_base + {len(SYNTHETIC_SIZES)} datasets sintéticos")


# =====================================================
# INSTALACIÓN DE LA EXTENSIÓN SQL
# =====================================================
def install_extension():
    sql_path = os.path.join(os.path.dirname(__file__), "sql", "kmeans_extension.sql")
    if not os.path.exists(sql_path):
        print(f"[!] No se encontró el archivo SQL: {sql_path}")
        return

    print(f"\n[+] Instalando extensión desde {sql_path}...")
    import subprocess
    psql_exe = r"C:\Program Files\PostgreSQL\18\bin\psql.exe"
    if not os.path.exists(psql_exe):
        psql_exe = "psql"
    result = subprocess.run(
        [psql_exe, "-U", DB_USER, "-h", DB_HOST, "-p", DB_PORT,
         "-d", DB_NAME, "-f", sql_path],
        capture_output=True, text=True,
        env={**os.environ, "PGPASSWORD": DB_PASSWORD}
    )
    if result.returncode == 0:
        print("[✓] Extensión instalada correctamente.")
    else:
        print(f"[!] Error instalando extensión:\n{result.stderr}")


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  CARGA DE WINE QUALITY DATASET → POSTGRESQL")
    print("=" * 65)

    create_database_if_not_exists()

    df_base = load_base_dataset()
    if df_base is None:
        exit(1)

    upload_datasets(df_base)
    install_extension()

    print("\n[✓] Proceso completado. Listo para ejecutar los benchmarks.")
