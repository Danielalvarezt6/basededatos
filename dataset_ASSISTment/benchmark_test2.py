"""
TEST 2 — Rendimiento variando número de atributos (Dimensionalidad Horizontal)
Replica el experimento 3.3.2 del artículo:
  Vallejo-Cabrera et al., Rev. Fac. Ing., Vol. 34, No. 74 (2025)

Configuración:
  - Registros: fijos en 21,000 (wine_quality_base)
  - K: 2 → 10
  - Atributos: 3, 4, 5, 6, 7, 8, 9, 10, 11 (subconjuntos progresivos del Wine Quality)
  - Herramienta: Extensión PL/Python en PostgreSQL

Salida: results/test2_extension_results.csv
"""

import json
import os
import time
import psycopg2
import pandas as pd

# =====================================================
# CONFIGURACIÓN
# =====================================================
DB_NAME     = "wine_quality"
DB_USER     = "postgres"
DB_PASSWORD = "danonino32"
DB_HOST     = "127.0.0.1"
DB_PORT     = "5432"

TABLE_NAME = "wine_quality_base"   # 21,000 registros
K_VALUES   = list(range(2, 11))    # K de 2 a 10
CSV_OUT    = "results/test2_extension_results.csv"

# Las 11 columnas del artículo; subconjuntos progresivos de 3 a 11
ALL_FEATURE_COLS = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide",
    "density", "ph", "sulphates", "alcohol"
]

# Subconjuntos de 3 a 11 atributos (se agregan de uno en uno)
ATTRIBUTE_SUBSETS = {
    n_attrs: ALL_FEATURE_COLS[:n_attrs]
    for n_attrs in range(3, 12)
}


def run_extension_benchmark():
    os.makedirs("results", exist_ok=True)
    results = []

    print("\n" + "=" * 70)
    print("TEST 2 — Extensión PL/Python: variando atributos y K (21K registros)")
    print("=" * 70)

    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
        host=DB_HOST, port=DB_PORT
    )
    conn.autocommit = True
    cur = conn.cursor()

    # Warm-up
    print("\n[~] Calentamiento del motor PL/Python...")
    try:
        cols_3 = ALL_FEATURE_COLS[:3]
        cur.execute("SELECT clustering.load_table_py(%s::TEXT, %s::TEXT[])", (TABLE_NAME, cols_3))
        cur.execute("SELECT clustering.preprocessing_py(%s::TEXT[])", (cols_3,))
        cur.execute("SELECT clustering.kmeans_py(3, 100, 42)")
        print("[~] Calentamiento completado.")
    except Exception as e:
        print(f"[!] Warning en calentamiento: {e}")

    total_runs = len(ATTRIBUTE_SUBSETS) * len(K_VALUES)
    run_n = 0

    for n_attrs, cols in ATTRIBUTE_SUBSETS.items():
        print(f"\n{'─'*60}")
        print(f"[+] {n_attrs} atributos: {cols}")

        # Carga y preprocesamiento para este subconjunto de atributos
        try:
            t_load0 = time.perf_counter()
            cur.execute("SELECT clustering.load_table_py(%s::TEXT, %s::TEXT[])", (TABLE_NAME, cols))
            cur.execute("SELECT clustering.preprocessing_py(%s::TEXT[])", (cols,))
            t_load1 = time.perf_counter()
            print(f"[✓] Carga + preprocesamiento: {t_load1 - t_load0:.4f} s")
        except Exception as e:
            print(f"[!] Error en preprocesamiento: {e}")
            continue

        for k in K_VALUES:
            run_n += 1
            print(f"\n  [{run_n}/{total_runs}] attrs={n_attrs} | k={k}", end=" ... ", flush=True)

            try:
                t0 = time.perf_counter()
                cur.execute(f"SELECT clustering.kmeans_py({k}, 300, 42)")
                raw = cur.fetchone()[0]
                t1 = time.perf_counter()

                payload = json.loads(raw)
                if not payload.get("ok"):
                    print(f"ERROR: {payload.get('error')}")
                    continue

                kmeans_time = payload["training_time_seconds"]
                total_time  = t1 - t0

                print(f"kmeans={kmeans_time:.4f}s | total={total_time:.4f}s | iters={payload['iterations']}")

                results.append({
                    "tool": "Extension",
                    "records": 21_000,
                    "k": k,
                    "attributes": n_attrs,
                    "kmeans_time_seconds": kmeans_time,
                    "total_time_seconds": total_time,
                    "iterations": payload["iterations"],
                    "inertia": payload["inertia"],
                })

            except Exception as e:
                print(f"ERROR: {e}")
                conn.autocommit = True
                continue

    cur.close()
    conn.close()

    if results:
        df = pd.DataFrame(results)
        df.to_csv(CSV_OUT, index=False)
        print(f"\n[✓] Resultados guardados en: {os.path.abspath(CSV_OUT)}")
        print(df.pivot_table(index="attributes", columns="k", values="kmeans_time_seconds").to_string())
    else:
        print("\n[!] No se generaron resultados.")

    return results


if __name__ == "__main__":
    run_extension_benchmark()
