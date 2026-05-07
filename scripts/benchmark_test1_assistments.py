"""
PRUEBA 1 — Rendimiento variando volumen de registros (Dimensionalidad Vertical)
Dataset: ASSISTments completo — 6.1M interacciones reales
Base de datos: assistments_clustering | Tabla: interacciones

Estrategia: para cada tamaño N se toma una muestra real (LIMIT N) de la tabla
de interacciones. Sin datos sintéticos.

Configuración:
  - Registros: 1K, 2K, 5K, 10K, 21K, 50K, 100K, 500K, 1M (todos reales)
  - Número de grupos (k): 2 → 10
  - Atributos: fijos en 11 columnas numéricas de la interacción
  - Herramienta: Extensión PL/Python en PostgreSQL

Salida: results/assistments/prueba1_extension.csv
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import psycopg2

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
from paths import RESULTS_ASSISTMENTS

# =====================================================
# CONFIGURACIÓN
# =====================================================
DB_NAME     = "assistments_clustering"
DB_USER     = "postgres"
DB_PASSWORD = os.environ.get("PGPASSWORD", "danonino32")
DB_HOST     = "127.0.0.1"
DB_PORT     = "5432"

TABLA_REAL  = "interacciones"
K_VALORES   = list(range(2, 11))
CSV_SALIDA  = str(RESULTS_ASSISTMENTS / "prueba1_extension.csv")

COLUMNAS = [
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

TAMANOS = [1_000, 2_000, 5_000, 10_000, 21_000, 50_000, 100_000, 500_000, 1_000_000]


def crear_muestra_real(cur, tamano: int) -> str:
    """
    Crea una tabla temporal con 'tamano' filas aleatorias de la tabla real.
    Usa setseed(0.42) — IDÉNTICO a sklearn y Weka (corrección P1).
    Devuelve el nombre de la tabla temporal.
    """
    nombre = f"_muestra_{tamano}"
    cols   = ", ".join(f'"{c}"' for c in COLUMNAS)
    # Semilla idéntica entre las tres herramientas para que reciban los mismos datos
    cur.execute("SELECT setseed(0.42)")
    cur.execute(f"""
        DROP TABLE IF EXISTS {nombre};
        CREATE TEMP TABLE {nombre} AS
        SELECT {cols}
        FROM {TABLA_REAL}
        ORDER BY random()
        LIMIT {tamano}
    """)
    return nombre


def ejecutar_benchmark():
    os.makedirs(RESULTS_ASSISTMENTS, exist_ok=True)
    resultados = []

    print("\n" + "=" * 70)
    print("PRUEBA 1 — Extensión PL/Python: variando registros (datos REALES)")
    print("Dataset: ASSISTments — 6.1M interacciones reales")
    print("=" * 70)

    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
        host=DB_HOST, port=DB_PORT
    )
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute(f"SELECT COUNT(*) FROM {TABLA_REAL}")
    total_real = cur.fetchone()[0]
    print(f"\n[✓] Filas reales disponibles en '{TABLA_REAL}': {total_real:,}")

    tamanos = [t for t in TAMANOS if t <= total_real]
    print(f"[✓] Tamaños a evaluar: {[f'{t:,}' for t in tamanos]}")

    # Calentamiento
    print("\n[~] Calentamiento del motor PL/Python...")
    try:
        tmp = crear_muestra_real(cur, 1_000)
        cur.execute("SELECT clustering.load_table_py(%s::TEXT, %s::TEXT[])", (tmp, COLUMNAS))
        cur.execute("SELECT clustering.preprocessing_py(%s::TEXT[])", (COLUMNAS,))
        cur.execute("SELECT clustering.kmeans_py(3, 50, 42)")
        print("[~] Calentamiento completado.")
    except Exception as e:
        print(f"[!] Advertencia en calentamiento: {e}")

    total_ejecuciones = len(tamanos) * len(K_VALORES)
    n_ejecucion = 0

    for tamano in tamanos:
        print(f"\n{'─' * 60}")
        print(f"[+] Muestra real de {tamano:,} filas (ORDER BY random() LIMIT {tamano:,})...")

        try:
            nombre_tmp = crear_muestra_real(cur, tamano)
            # Tiempo de carga interna (sin transferencia de red — datos ya en la BD)
            t0_carga = time.perf_counter()
            cur.execute("SELECT clustering.load_table_py(%s::TEXT, %s::TEXT[])",
                        (nombre_tmp, COLUMNAS))
            cur.execute("SELECT clustering.preprocessing_py(%s::TEXT[])", (COLUMNAS,))
            t_carga = time.perf_counter() - t0_carga
            print(f"[✓] Carga interna + normalización: {t_carga:.4f} s")
        except Exception as e:
            print(f"[!] Error con {tamano:,} registros: {e}")
            continue

        for k in K_VALORES:
            n_ejecucion += 1
            print(f"\n  [{n_ejecucion}/{total_ejecuciones}] registros={tamano:,} | grupos k={k}",
                  end=" ... ", flush=True)

            try:
                t0 = time.perf_counter()
                cur.execute("SELECT clustering.kmeans_py(%s, 300, 42)", (k,))
                raw = cur.fetchone()[0]
                t_total = time.perf_counter() - t0

                payload = json.loads(raw)
                if not payload.get("ok"):
                    print(f"ERROR: {payload.get('error')}")
                    continue

                t_kmeans = payload["training_time_seconds"]
                t_respuesta = t_carga + t_total   # tiempo de respuesta total al usuario
                print(f"kmeans={t_kmeans:.4f}s | carga={t_carga:.4f}s | "
                      f"respuesta={t_respuesta:.4f}s | iteraciones={payload['iterations']}")

                resultados.append({
                    "herramienta":         "Extensión PostgreSQL",
                    "registros":           tamano,
                    "num_grupos":          k,
                    "num_atributos":       len(COLUMNAS),
                    "tiempo_carga_s":      round(t_carga, 6),
                    # P4: tiempo_kmeans_s = llamada SQL completa (equivalente a sklearn/Weka)
                    "tiempo_kmeans_s":         round(t_total, 6),
                    "tiempo_kmeans_interno_s": round(t_kmeans, 6),
                    "tiempo_total_s":          round(t_total, 6),
                    "tiempo_respuesta_s":      round(t_respuesta, 6),
                    "iteraciones":             payload["iterations"],
                    "inercia_wcss":            round(payload["inertia"], 4),
                })

            except Exception as e:
                print(f"ERROR: {e}")
                conn.autocommit = True
                continue

    cur.close()
    conn.close()

    if resultados:
        df = pd.DataFrame(resultados)
        df.to_csv(CSV_SALIDA, index=False, encoding="utf-8")
        print(f"\n[✓] Resultados guardados en: {os.path.abspath(CSV_SALIDA)}")
        pivot = df.pivot_table(index="registros", columns="num_grupos",
                               values="tiempo_kmeans_s")
        pivot.columns.name = "k (grupos)"
        pivot.index.name   = "Registros"
        print("\nTiempo K-Means (segundos) — Extensión PL/Python:")
        print(pivot.to_string())
    else:
        print("\n[!] No se generaron resultados.")


if __name__ == "__main__":
    ejecutar_benchmark()
