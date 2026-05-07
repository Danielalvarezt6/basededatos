"""
Benchmark Python scikit-learn KMeans — Dataset ASSISTments (6.1M reales)
  - Prueba 1: variando número de registros (1K → 1M, todos reales)
  - Prueba 2: variando número de atributos (100K filas reales fijas)

Salidas:
  results/assistments/prueba1_sklearn.csv
  results/assistments/prueba2_sklearn.csv
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
from paths import RESULTS_ASSISTMENTS

CSV_PRUEBA1_SKLEARN = str(RESULTS_ASSISTMENTS / "prueba1_sklearn.csv")
CSV_PRUEBA2_SKLEARN = str(RESULTS_ASSISTMENTS / "prueba2_sklearn.csv")

# =====================================================
# CONFIGURACIÓN
# =====================================================
DB_NAME     = "assistments_clustering"
DB_USER     = "postgres"
DB_PASSWORD = os.environ.get("PGPASSWORD", "danonino32")
DB_HOST     = "127.0.0.1"
DB_PORT     = "5432"

TABLA_REAL  = "interacciones"
TAMANOS     = [1_000, 2_000, 5_000, 10_000, 21_000, 50_000, 100_000, 500_000, 1_000_000]
K_VALORES   = list(range(2, 11))
N_PRUEBA2   = 100_000

TODAS_LAS_COLUMNAS = [
    "ms_first_response",
    "hint_count",
    "attempt_count",
    "correct",
    "original",
    "bottom_hint",
    "overlap_time",
    "Average_confidence(FRUSTRATED)",
    "Average_confidence(CONFUSED)",
    "Average_confidence(CONCENTRATING)",
    "Average_confidence(BORED)",
]

SUBCONJUNTOS_ATRIBUTOS = {n: TODAS_LAS_COLUMNAS[:n] for n in [3, 5, 7, 9, 11]}


# =====================================================
# UTILIDADES
# =====================================================
def crear_tabla_muestra(cur, conn, tamano: int, cols: list) -> str:
    """
    Materializa N filas aleatorias en una tabla temporal (no se cronometra).
    Usa setseed(0.42) — IDÉNTICO a la extensión y Weka (corrección P1).
    Devuelve el nombre de la tabla para que el benchmark la lea después.
    """
    nombre = f"_muestra_{tamano}"
    col_clause = ", ".join(f'"{c}"' for c in cols)
    # Semilla idéntica a la extensión: setseed(0.42) — NO setseed(0.000000042)
    cur.execute("SELECT setseed(0.42)")
    cur.execute(f"""
        DROP TABLE IF EXISTS {nombre};
        CREATE TEMP TABLE {nombre} AS
        SELECT {col_clause}
        FROM {TABLA_REAL}
        ORDER BY random()
        LIMIT {tamano}
    """)
    return nombre


def normalizar(df: pd.DataFrame) -> np.ndarray:
    return MinMaxScaler().fit_transform(df.values.astype(float))


def correr_kmeans(X: np.ndarray, k: int) -> tuple[float, int, float]:
    modelo = KMeans(n_clusters=k, init="k-means++", max_iter=300,
                    random_state=42, n_init=1)
    t0 = time.perf_counter()
    modelo.fit(X)
    return time.perf_counter() - t0, modelo.n_iter_, modelo.inertia_


# =====================================================
# PRUEBA 1 — Variando registros
# =====================================================
def prueba1_sklearn():
    os.makedirs(RESULTS_ASSISTMENTS, exist_ok=True)
    resultados = []

    print("\n" + "=" * 70)
    print("PRUEBA 1 — sklearn KMeans: variando registros (datos REALES)")
    print("Dataset: ASSISTments — 6.1M interacciones reales")
    print("=" * 70)

    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
        host=DB_HOST, port=DB_PORT
    )
    conn.autocommit = True
    cur = conn.cursor()

    total_ejecuciones = len(TAMANOS) * len(K_VALORES)
    n_ejecucion = 0

    for tamano in TAMANOS:
        print(f"\n{'─' * 60}")
        print(f"[+] Preparando muestra de {tamano:,} filas (ORDER BY random — no se mide)...")
        try:
            # Crear tabla temporal con misma semilla que la extensión — NO se cronometra
            nombre_tmp = crear_tabla_muestra(cur, conn, tamano, TODAS_LAS_COLUMNAS)
            print(f"[✓] Tabla temporal lista: {nombre_tmp}")
        except Exception as e:
            print(f"[!] Error preparando muestra: {e}")
            continue

        # === MEDICIÓN: leer desde tabla temp + normalizar (misma condición que extensión) ===
        try:
            t0_descarga = time.perf_counter()
            df = pd.read_sql(f'SELECT * FROM {nombre_tmp}', conn)
            t_descarga = time.perf_counter() - t0_descarga
        except Exception as e:
            print(f"[!] Error leyendo muestra: {e}")
            continue

        t0_norm = time.perf_counter()
        X = normalizar(df)
        t_norm = time.perf_counter() - t0_norm
        t_carga = t_descarga + t_norm
        print(f"[✓] Descarga tabla temp: {t_descarga:.4f}s | Normalización: {t_norm:.4f}s | Total carga: {t_carga:.4f}s")

        for k in K_VALORES:
            n_ejecucion += 1
            print(f"\n  [{n_ejecucion}/{total_ejecuciones}] registros={tamano:,} | k={k}",
                  end=" ... ", flush=True)
            try:
                t_kmeans, iters, inercia = correr_kmeans(X, k)
                t_respuesta = t_carga + t_kmeans
                print(f"kmeans={t_kmeans:.4f}s | carga={t_carga:.4f}s | respuesta={t_respuesta:.4f}s | iters={iters}")
                resultados.append({
                    "herramienta":         "Python sklearn",
                    "registros":           tamano,
                    "num_grupos":          k,
                    "num_atributos":       len(TODAS_LAS_COLUMNAS),
                    "tiempo_carga_s":      round(t_carga, 6),
                    "tiempo_kmeans_s":     round(t_kmeans, 6),
                    "tiempo_total_s":      round(t_norm + t_kmeans, 6),
                    "tiempo_respuesta_s":  round(t_respuesta, 6),
                    "iteraciones":         iters,
                    "inercia_wcss":        round(inercia, 4),
                })
            except Exception as e:
                print(f"ERROR: {e}")

    if resultados:
        pd.DataFrame(resultados).to_csv(
            CSV_PRUEBA1_SKLEARN, index=False, encoding="utf-8")
        print(f"\n[✓] Guardado: {CSV_PRUEBA1_SKLEARN}")

    cur.close()
    conn.close()
    return resultados


# =====================================================
# PRUEBA 2 — Variando atributos
# =====================================================
def prueba2_sklearn():
    os.makedirs(RESULTS_ASSISTMENTS, exist_ok=True)
    resultados = []

    print("\n" + "=" * 70)
    print(f"PRUEBA 2 — sklearn KMeans: variando atributos | {N_PRUEBA2:,} filas reales")
    print("=" * 70)

    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
        host=DB_HOST, port=DB_PORT
    )
    conn.autocommit = True
    cur = conn.cursor()

    # Crear tabla temporal fija (NO se cronometra)
    print(f"\n[+] Preparando muestra fija de {N_PRUEBA2:,} filas (no se mide)...")
    try:
        nombre_tmp = crear_tabla_muestra(cur, conn, N_PRUEBA2, TODAS_LAS_COLUMNAS)
        print(f"[✓] Tabla temporal lista: {nombre_tmp}")
    except Exception as e:
        print(f"[!] Error: {e}")
        conn.close()
        return []

    # Leer de tabla temp (se mide una vez) y reutilizar para todos los subconjuntos
    t0_desc = time.perf_counter()
    df_completo = pd.read_sql(f'SELECT * FROM {nombre_tmp}', conn)
    t_descarga_p2 = time.perf_counter() - t0_desc
    print(f"[✓] Descarga tabla temp: {t_descarga_p2:.4f} s")

    total_ejecuciones = len(SUBCONJUNTOS_ATRIBUTOS) * len(K_VALORES)
    n_ejecucion = 0

    for n_attrs, cols in SUBCONJUNTOS_ATRIBUTOS.items():
        print(f"\n{'─' * 60}")
        print(f"[+] Normalizando {n_attrs} atributos...")
        t0_norm = time.perf_counter()
        X = normalizar(df_completo[cols])
        t_norm = time.perf_counter() - t0_norm
        t_carga = t_descarga_p2 + t_norm
        print(f"[✓] Normalización: {t_norm:.4f}s | Carga total: {t_carga:.4f}s")

        for k in K_VALORES:
            n_ejecucion += 1
            print(f"\n  [{n_ejecucion}/{total_ejecuciones}] atributos={n_attrs} | k={k}",
                  end=" ... ", flush=True)
            try:
                t_kmeans, iters, inercia = correr_kmeans(X, k)
                t_respuesta = t_carga + t_kmeans
                print(f"kmeans={t_kmeans:.4f}s | carga={t_carga:.4f}s | respuesta={t_respuesta:.4f}s | iters={iters}")
                resultados.append({
                    "herramienta":         "Python sklearn",
                    "registros":           N_PRUEBA2,
                    "num_grupos":          k,
                    "num_atributos":       n_attrs,
                    "tiempo_carga_s":      round(t_carga, 6),
                    "tiempo_kmeans_s":     round(t_kmeans, 6),
                    "tiempo_total_s":      round(t_norm + t_kmeans, 6),
                    "tiempo_respuesta_s":  round(t_respuesta, 6),
                    "iteraciones":         iters,
                    "inercia_wcss":        round(inercia, 4),
                })
            except Exception as e:
                print(f"ERROR: {e}")

    if resultados:
        pd.DataFrame(resultados).to_csv(
            CSV_PRUEBA2_SKLEARN, index=False, encoding="utf-8")
        print(f"\n[✓] Guardado: {CSV_PRUEBA2_SKLEARN}")

    cur.close()
    conn.close()
    return resultados


if __name__ == "__main__":
    prueba1_sklearn()
    prueba2_sklearn()
