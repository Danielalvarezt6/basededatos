"""
Agrega el dataset ASSISTments crudo a nivel de estudiante
y carga el resultado en PostgreSQL para los benchmarks.

Genera 11 variables de comportamiento por estudiante:
  avg_response_time, correct_rate, avg_attempts, problems_completed,
  hint_rate, skill_diversity, night_activity, weekend_activity,
  persistence_score, mastery_estimate, session_length

Base de datos destino: assistments_clustering
Tabla base:            student_profiles  (~N estudiantes únicos)
Tablas sintéticas:     student_synth_1000 … student_synth_1000000
"""

import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
import subprocess

# =====================================================
# CONFIGURACIÓN
# =====================================================
CSV_PATH    = r"C:\Users\deat_\Downloads\fuentescartelcecen\dataset_ASSISTment\dataset.csv"

DB_USER     = "postgres"
DB_PASSWORD = "danonino32"
DB_HOST     = "127.0.0.1"
DB_PORT     = "5432"
DB_NAME     = "assistments_clustering"

PSQL_EXE    = r"C:\Program Files\PostgreSQL\18\bin\psql.exe"
SQL_EXT     = os.path.join(os.path.dirname(__file__), "sql", "kmeans_extension.sql")

# 11 features numéricas (sin student_id)
FEATURE_COLS = [
    "avg_response_time",
    "correct_rate",
    "avg_attempts",
    "problems_completed",
    "hint_rate",
    "skill_diversity",
    "night_activity",
    "weekend_activity",
    "persistence_score",
    "mastery_estimate",
    "session_length",
]

# Tamaños sintéticos para Test 1
SYNTHETIC_SIZES = [1_000, 2_000, 5_000, 10_000, 21_000, 50_000, 100_000, 500_000, 1_000_000]


# =====================================================
# UTILIDADES DE BASE DE DATOS
# =====================================================
def get_engine(db_name: str = DB_NAME):
    return create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{db_name}",
        pool_pre_ping=True,
    )


def crear_base_si_no_existe():
    engine = get_engine("postgres")
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        existe = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :db"), {"db": DB_NAME}
        ).fetchone()
        if not existe:
            conn.execute(text(f'CREATE DATABASE "{DB_NAME}"'))
            print(f"[+] Base de datos '{DB_NAME}' creada.")
        else:
            print(f"[~] Base de datos '{DB_NAME}' ya existe.")


# =====================================================
# AGREGACIÓN POR ESTUDIANTE
# =====================================================
def mastery_estimate(group: pd.DataFrame) -> float:
    """Tasa de acierto en el último 20 % de problemas del estudiante."""
    group_sorted = group.sort_values("start_time")
    n = max(1, len(group_sorted) // 5)
    return float(group_sorted["correct"].iloc[-n:].mean())


def agregar_por_estudiante(csv_path: str) -> pd.DataFrame:
    print(f"\n[+] Leyendo CSV: {csv_path}")
    df = pd.read_csv(
        csv_path,
        usecols=["user_id", "correct", "hint_count", "attempt_count",
                 "ms_first_response", "skill_id", "start_time", "end_time"],
        low_memory=False,
    )
    # Convertir tipos después de cargar para evitar conflictos con NaN
    df["user_id"]           = pd.to_numeric(df["user_id"],           errors="coerce")
    df["correct"]           = pd.to_numeric(df["correct"],           errors="coerce")
    df["hint_count"]        = pd.to_numeric(df["hint_count"],        errors="coerce")
    df["attempt_count"]     = pd.to_numeric(df["attempt_count"],     errors="coerce")
    df["ms_first_response"] = pd.to_numeric(df["ms_first_response"], errors="coerce")
    df["skill_id"]          = pd.to_numeric(df["skill_id"],          errors="coerce")
    df["start_time"]        = pd.to_datetime(df["start_time"],       errors="coerce")
    df["end_time"]          = pd.to_datetime(df["end_time"],         errors="coerce")

    print(f"[✓] Filas cargadas: {len(df):,}")
    df = df.dropna(subset=["user_id", "ms_first_response", "correct"])
    df = df[df["ms_first_response"] > 0]
    print(f"[✓] Filas válidas tras filtrar: {len(df):,}")

    print("[+] Agregando por estudiante (user_id)...")

    # Variables temporales auxiliares
    df["used_hint"]   = (df["hint_count"] > 0).astype(int)
    df["hour"]        = df["start_time"].dt.hour
    df["is_night"]    = df["hour"].apply(lambda h: 1 if h >= 20 or h < 6 else 0)
    df["is_weekend"]  = df["start_time"].dt.dayofweek.apply(lambda d: 1 if d >= 5 else 0)
    df["duration_s"]  = (df["end_time"] - df["start_time"]).dt.total_seconds().clip(lower=0, upper=3600)

    # persistence_score: intentos promedio en problemas incorrectos
    incorrect = df[df["correct"] == 0].groupby("user_id")["attempt_count"].mean().rename("persistence_score")

    # mastery_estimate por estudiante
    mastery = df.groupby("user_id").apply(mastery_estimate, include_groups=False).rename("mastery_estimate")

    agg = df.groupby("user_id").agg(
        avg_response_time  = ("ms_first_response", "mean"),
        correct_rate       = ("correct",           "mean"),
        avg_attempts       = ("attempt_count",     "mean"),
        problems_completed = ("user_id",           "count"),
        hint_rate          = ("used_hint",         "mean"),
        skill_diversity    = ("skill_id",          "nunique"),
        night_activity     = ("is_night",          "mean"),
        weekend_activity   = ("is_weekend",        "mean"),
        session_length     = ("duration_s",        "mean"),
    ).reset_index()

    agg = agg.merge(incorrect, on="user_id", how="left")
    agg = agg.merge(mastery,   on="user_id", how="left")

    agg["persistence_score"] = agg["persistence_score"].fillna(agg["avg_attempts"])
    agg["mastery_estimate"]  = agg["mastery_estimate"].fillna(agg["correct_rate"])

    # Eliminar estudiantes con muy pocos problemas (ruido)
    agg = agg[agg["problems_completed"] >= 5].reset_index(drop=True)

    print(f"[✓] Estudiantes únicos con >= 5 problemas: {len(agg):,}")
    return agg


# =====================================================
# GENERACIÓN DE DATASETS SINTÉTICOS
# =====================================================
def generate_synthetic(df_base: pd.DataFrame, size: int, rng: np.random.Generator) -> pd.DataFrame:
    if size <= len(df_base):
        seed = int(rng.integers(0, 10**9))
        return pd.DataFrame(
            df_base[FEATURE_COLS].sample(n=size, random_state=seed).values,
            columns=FEATURE_COLS,
        )
    n_repeats = size // len(df_base) + 1
    df_big = pd.concat([df_base[FEATURE_COLS]] * n_repeats, ignore_index=True).iloc[:size].copy()
    noise_scale = np.asarray(df_base[FEATURE_COLS].std()) * 0.01
    noise = rng.normal(0, noise_scale, size=df_big.shape)
    df_big = pd.DataFrame(df_big.to_numpy() + noise, columns=FEATURE_COLS).clip(lower=0)
    return df_big.reset_index(drop=True)


# =====================================================
# CARGA EN POSTGRESQL
# =====================================================
def subir_datasets(df_base: pd.DataFrame):
    engine = get_engine()
    rng = np.random.default_rng(42)

    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS clustering"))
        conn.commit()

    # Tabla base (perfiles reales de estudiantes)
    print(f"\n[+] Subiendo 'student_profiles' ({len(df_base):,} filas)...")
    df_base[FEATURE_COLS].to_sql(
        "student_profiles", engine, if_exists="replace", index=False, method="multi"
    )
    print(f"[✓] 'student_profiles' lista.")

    for size in SYNTHETIC_SIZES:
        tabla = f"student_synth_{size}"
        print(f"\n[+] Generando '{tabla}' ({size:,} filas)...")
        df_s = generate_synthetic(df_base, size, rng)
        df_s.to_sql(tabla, engine, if_exists="replace", index=False,
                    method="multi", chunksize=10_000)
        print(f"[✓] '{tabla}' lista.")

    print(f"\n[✓] Todos los datasets cargados en '{DB_NAME}'.")


# =====================================================
# INSTALACIÓN DE LA EXTENSIÓN
# =====================================================
def instalar_extension():
    if not os.path.exists(SQL_EXT):
        print(f"[!] No se encontró: {SQL_EXT}")
        return
    print(f"\n[+] Instalando extensión PL/Python...")
    env = {**os.environ, "PGPASSWORD": DB_PASSWORD}

    # Habilitar plpython3u primero
    subprocess.run(
        [PSQL_EXE, "-U", DB_USER, "-h", DB_HOST, "-p", DB_PORT,
         "-d", DB_NAME, "-c", "CREATE EXTENSION IF NOT EXISTS plpython3u;"],
        capture_output=True, text=True, env=env
    )

    result = subprocess.run(
        [PSQL_EXE, "-U", DB_USER, "-h", DB_HOST, "-p", DB_PORT,
         "-d", DB_NAME, "-f", SQL_EXT],
        capture_output=True, text=True, env=env
    )
    if result.returncode == 0:
        print("[✓] Extensión instalada correctamente.")
    else:
        print(f"[!] Error:\n{result.stderr[:500]}")


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  AGREGACIÓN ASSISTments -> PERFILES DE ESTUDIANTES -> POSTGRESQL")
    print("=" * 65)

    crear_base_si_no_existe()
    df_estudiantes = agregar_por_estudiante(CSV_PATH)

    # Guardar CSV intermedio para inspección
    out_csv = os.path.join(os.path.dirname(CSV_PATH), "student_profiles.csv")
    df_estudiantes.to_csv(out_csv, index=False)
    print(f"\n[✓] Perfiles guardados en: {out_csv}")
    print(df_estudiantes[FEATURE_COLS].describe().round(3).to_string())

    subir_datasets(df_estudiantes)
    instalar_extension()

    print("\n[✓] Todo listo. Ejecuta benchmark_test1_assistments.py para iniciar.")
