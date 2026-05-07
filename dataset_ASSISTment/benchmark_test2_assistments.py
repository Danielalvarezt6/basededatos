"""
PRUEBA 2 — Rendimiento variando número de atributos (Dimensionalidad Horizontal)
Dataset: ASSISTments completo — 6.1M interacciones reales
Base de datos: assistments_clustering | Tabla: interacciones

Configuración:
  - Registros: fijos en 100K filas reales
  - Número de grupos (k): 2 → 10
  - Atributos: de 3 a 11 columnas numéricas de la interacción
  - Herramienta: Extensión PL/Python en PostgreSQL

Salida: results/assistments/prueba2_extension.csv
"""

import json
import os
import time
import psycopg2
import pandas as pd

# =====================================================
# CONFIGURACIÓN
# =====================================================
DB_NAME     = "assistments_clustering"
DB_USER     = "postgres"
DB_PASSWORD = "danonino32"
DB_HOST     = "127.0.0.1"
DB_PORT     = "5432"

TABLA_REAL    = "interacciones"
N_REGISTROS   = 100_000   # filas reales fijas para Test 2
K_VALORES     = list(range(2, 11))
CSV_SALIDA    = "results/assistments/prueba2_extension.csv"

# 11 columnas numéricas reales del dataset ASSISTments
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

# Subconjuntos: 3, 5, 7, 9, 11 atributos
SUBCONJUNTOS = [TODAS_LAS_COLUMNAS[:n] for n in [3, 5, 7, 9, 11]]


def ejecutar_benchmark():
    os.makedirs("results/assistments", exist_ok=True)
    resultados = []

    print("\n" + "=" * 70)
    print("PRUEBA 2 — Extensión PL/Python: variando atributos (datos REALES)")
    print(f"Dataset: ASSISTments | {N_REGISTROS:,} filas reales fijas")
    print("=" * 70)

    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
        host=DB_HOST, port=DB_PORT
    )
    conn.autocommit = True
    cur = conn.cursor()

    # Crear tabla temporal fija con N_REGISTROS filas reales
    print(f"\n[+] Creando muestra fija de {N_REGISTROS:,} filas reales...")
    cols_all = ", ".join(f'"{c}"' for c in TODAS_LAS_COLUMNAS)
    cur.execute(f"SELECT setseed(0.42)")
    cur.execute(f"""
        DROP TABLE IF EXISTS _muestra_prueba2;
        CREATE TEMP TABLE _muestra_prueba2 AS
        SELECT {cols_all}
        FROM {TABLA_REAL}
        ORDER BY random()
        LIMIT {N_REGISTROS}
    """)
    print(f"[✓] Muestra fija creada.")

    total_ejecuciones = len(SUBCONJUNTOS) * len(K_VALORES)
    n_ejecucion = 0

    for columnas in SUBCONJUNTOS:
        n_cols = len(columnas)
        print(f"\n{'─' * 60}")
        print(f"[+] Cargando {N_REGISTROS:,} filas con {n_cols} atributos: {columnas}")

        try:
            t0_carga = time.perf_counter()
            cur.execute("SELECT clustering.load_table_py(%s::TEXT, %s::TEXT[])",
                        ("_muestra_prueba2", columnas))
            cur.execute("SELECT clustering.preprocessing_py(%s::TEXT[])", (columnas,))
            t_carga = time.perf_counter() - t0_carga
            print(f"[✓] Carga interna + normalización: {t_carga:.4f} s")
        except Exception as e:
            print(f"[!] Error con {n_cols} atributos: {e}")
            continue

        for k in K_VALORES:
            n_ejecucion += 1
            print(f"\n  [{n_ejecucion}/{total_ejecuciones}] atributos={n_cols} | grupos k={k}",
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
                t_respuesta = t_carga + t_total
                print(f"kmeans={t_kmeans:.4f}s | carga={t_carga:.4f}s | "
                      f"respuesta={t_respuesta:.4f}s | iteraciones={payload['iterations']}")

                resultados.append({
                    "herramienta":         "Extensión PostgreSQL",
                    "registros":           N_REGISTROS,
                    "num_grupos":          k,
                    "num_atributos":       n_cols,
                    "tiempo_carga_s":          round(t_carga, 6),
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
        pivot = df.pivot_table(index="num_atributos", columns="num_grupos",
                               values="tiempo_kmeans_s")
        pivot.columns.name = "k (grupos)"
        pivot.index.name   = "Atributos"
        print("\nTiempo K-Means (s) — Extensión PL/Python (por atributos):")
        print(pivot.to_string())
    else:
        print("\n[!] No se generaron resultados.")


if __name__ == "__main__":
    ejecutar_benchmark()
