"""
Benchmark de Weka — ejecuta SimpleKMeans desde línea de comandos
y recolecta los tiempos para Test 1 y Test 2.

Requisitos:
  - Java instalado y en el PATH del sistema
  - weka.jar descargado. Ruta indicada en WEKA_JAR_PATH.
    Descarga: https://sourceforge.net/projects/weka/files/weka-3-8/

Salidas:
  results/test1_weka_results.csv
  results/test2_weka_results.csv
"""

import os
import re
import subprocess
import tempfile
import time
import pandas as pd
import numpy as np
import psycopg2

# =====================================================
# CONFIGURACIÓN
# =====================================================
WEKA_JAR_PATH = r"C:\Program Files\Weka-3-8-7\weka.jar"
JAVA_EXE_PATH = r"C:\Program Files\Weka-3-8-7\jre\jre-25.0.2-full\bin\java.exe"

DB_NAME     = "wine_quality"
DB_USER     = "postgres"
DB_PASSWORD = "danonino32"
DB_HOST     = "127.0.0.1"
DB_PORT     = "5432"

# Test 1
SYNTHETIC_SIZES = [1_000, 2_000, 5_000, 10_000, 21_000, 50_000, 100_000, 500_000, 1_000_000]
K_VALUES        = list(range(2, 11))

ALL_FEATURE_COLS = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide",
    "density", "ph", "sulphates", "alcohol"
]

ATTRIBUTE_SUBSETS = {n: ALL_FEATURE_COLS[:n] for n in range(3, 12)}


# =====================================================
# UTILIDADES
# =====================================================

def fetch_table_as_df(table_name: str, cols: list) -> pd.DataFrame:
    """Descarga una tabla de PostgreSQL como DataFrame."""
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
        host=DB_HOST, port=DB_PORT
    )
    col_clause = ", ".join(f'"{c}"' for c in cols)
    df = pd.read_sql(f"SELECT {col_clause} FROM {table_name}", conn)
    conn.close()
    return df


def df_to_arff(df: pd.DataFrame, relation_name: str = "wine") -> str:
    """Convierte un DataFrame a formato ARFF (requerido por Weka)."""
    lines = [f"@RELATION {relation_name}", ""]
    for col in df.columns:
        lines.append(f"@ATTRIBUTE {col} NUMERIC")
    lines += ["", "@DATA"]
    for _, row in df.iterrows():
        lines.append(",".join(str(round(v, 6)) for v in row))
    return "\n".join(lines)


def run_weka_kmeans(arff_path: str, k: int, max_iter: int = 500, seed: int = 10) -> float | None:
    """
    Ejecuta Weka SimpleKMeans sobre un archivo ARFF y retorna el tiempo en segundos.
    Devuelve None si hay error.

    Parámetros según configuración por defecto de Weka (GUI SimpleKMeans):
      - initializationMethod: Random (Weka default, distinto de k-means++)
      - maxIterations: 500
      - seed: 10
      - distanceFunction: EuclideanDistance
    """
    if not os.path.exists(WEKA_JAR_PATH):
        raise FileNotFoundError(
            f"No se encontró weka.jar en: {WEKA_JAR_PATH}\n"
            "Descárgalo de: https://sourceforge.net/projects/weka/files/weka-3-8/"
        )

    java_exe = JAVA_EXE_PATH if os.path.exists(JAVA_EXE_PATH) else "java"
    cmd = [
        java_exe, "-Xmx4g",
        "-cp", WEKA_JAR_PATH,
        "weka.clusterers.SimpleKMeans",
        "-N", str(k),
        "-I", str(max_iter),
        "-S", str(seed),
        # Sin -init → usa Random (equivalente a initializationMethod=Random de la GUI)
        "-t", arff_path,
    ]

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"    [!] Weka error (k={k}): {result.stderr[:200]}")
        return None

    return elapsed


# =====================================================
# TEST 1 — Variando registros
# =====================================================
def run_test1_weka():
    os.makedirs("results", exist_ok=True)
    results = []

    print("\n" + "=" * 70)
    print("TEST 1 — Weka: variando registros y K")
    print("=" * 70)

    if not os.path.exists(WEKA_JAR_PATH):
        print(f"\n[!] weka.jar no encontrado en: {WEKA_JAR_PATH}")
        print("    Descarga Weka desde: https://sourceforge.net/projects/weka/files/weka-3-8/")
        print("    Luego ajusta WEKA_JAR_PATH en este script.")
        return []

    total_runs = len(SYNTHETIC_SIZES) * len(K_VALUES)
    run_n = 0

    for size in SYNTHETIC_SIZES:
        table_name = f"wine_synth_{size}"
        print(f"\n{'─'*60}")
        print(f"[+] Descargando '{table_name}' ({size:,} filas)...")

        try:
            df = fetch_table_as_df(table_name, ALL_FEATURE_COLS)
        except Exception as e:
            print(f"[!] Error descargando {table_name}: {e}")
            continue

        # Escribir ARFF temporal (se reutiliza para todos los K de este tamaño)
        with tempfile.NamedTemporaryFile(suffix=".arff", mode="w",
                                        delete=False, encoding="utf-8") as f:
            arff_path = f.name
            f.write(df_to_arff(df, relation_name=f"wine_{size}"))

        print(f"[✓] ARFF escrito: {arff_path}")

        for k in K_VALUES:
            run_n += 1
            print(f"\n  [{run_n}/{total_runs}] size={size:,} | k={k}", end=" ... ", flush=True)

            elapsed = run_weka_kmeans(arff_path, k, max_iter=500, seed=10)
            if elapsed is None:
                continue

            print(f"tiempo={elapsed:.4f}s")
            results.append({
                "tool": "Weka",
                "records": size,
                "k": k,
                "attributes": len(ALL_FEATURE_COLS),
                "total_time_seconds": elapsed,
            })

        os.unlink(arff_path)

    if results:
        df_out = pd.DataFrame(results)
        df_out.to_csv("results/test1_weka_results.csv", index=False)
        print(f"\n[✓] Resultados Weka Test 1 guardados en results/test1_weka_results.csv")
    return results


# =====================================================
# TEST 2 — Variando atributos
# =====================================================
def run_test2_weka():
    os.makedirs("results", exist_ok=True)
    results = []

    print("\n" + "=" * 70)
    print("TEST 2 — Weka: variando atributos y K (21K registros)")
    print("=" * 70)

    if not os.path.exists(WEKA_JAR_PATH):
        print(f"\n[!] weka.jar no encontrado en: {WEKA_JAR_PATH}")
        return []

    print("\n[+] Descargando wine_quality_base (21,000 filas)...")
    try:
        df_full = fetch_table_as_df("wine_quality_base", ALL_FEATURE_COLS)
    except Exception as e:
        print(f"[!] Error: {e}")
        return []

    total_runs = len(ATTRIBUTE_SUBSETS) * len(K_VALUES)
    run_n = 0

    for n_attrs, cols in ATTRIBUTE_SUBSETS.items():
        print(f"\n{'─'*60}")
        df_sub = pd.DataFrame(df_full[cols].values, columns=cols)

        with tempfile.NamedTemporaryFile(suffix=".arff", mode="w",
                                        delete=False, encoding="utf-8") as f:
            arff_path = f.name
            f.write(df_to_arff(df_sub, relation_name=f"wine_attrs{n_attrs}"))

        print(f"[+] {n_attrs} atributos | ARFF: {arff_path}")

        for k in K_VALUES:
            run_n += 1
            print(f"\n  [{run_n}/{total_runs}] attrs={n_attrs} | k={k}", end=" ... ", flush=True)

            elapsed = run_weka_kmeans(arff_path, k, max_iter=500, seed=10)
            if elapsed is None:
                continue

            print(f"tiempo={elapsed:.4f}s")
            results.append({
                "tool": "Weka",
                "records": 21_000,
                "k": k,
                "attributes": n_attrs,
                "total_time_seconds": elapsed,
            })

        os.unlink(arff_path)

    if results:
        df_out = pd.DataFrame(results)
        df_out.to_csv("results/test2_weka_results.csv", index=False)
        print(f"\n[✓] Resultados Weka Test 2 guardados en results/test2_weka_results.csv")
    return results


if __name__ == "__main__":
    run_test1_weka()
    run_test2_weka()
