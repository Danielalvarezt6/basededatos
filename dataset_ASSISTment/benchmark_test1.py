"""
TEST 1 — Rendimiento variando volumen de registros (Dimensionalidad Vertical)
Replica el experimento 3.3.1 del artículo:
  Vallejo-Cabrera et al., Rev. Fac. Ing., Vol. 34, No. 74 (2025)

Configuración:
  - Registros: 1K, 2K, 5K, 10K, 21K, 50K, 100K, 500K, 1M (sintéticos)
  - K: 2 → 10
  - Atributos: fijos en 11 (todas las variables fisicoquímicas del Wine Quality)
  - Herramienta: Extensión PL/Python en PostgreSQL

Salida: results/test1_extension_results.csv
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

# Nombres de las tablas sintéticas (creadas por cargar_wine_quality.py)
SYNTHETIC_SIZES = [1_000, 2_000, 5_000, 10_000, 21_000, 50_000, 100_000, 500_000, 1_000_000]
K_VALUES        = list(range(2, 11))   # K de 2 a 10
CSV_OUT         = "results/test1_extension_results.csv"

FEATURE_COLS = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide",
    "density", "ph", "sulphates", "alcohol"
]


def run_extension_benchmark():
    os.makedirs("results", exist_ok=True)
    results = []

    print("\n" + "=" * 70)
    print("TEST 1 — Extensión PL/Python: variando registros y K")
    print("=" * 70)

    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
        host=DB_HOST, port=DB_PORT
    )
    conn.autocommit = True
    cur = conn.cursor()

    # Warm-up con tabla pequeña
    print("\n[~] Calentamiento del motor PL/Python...")
    try:
        cur.execute("SELECT clustering.load_table_py(%s::TEXT, %s::TEXT[])", ("wine_synth_1000", FEATURE_COLS))
        cur.execute("SELECT clustering.preprocessing_py(%s::TEXT[])", (FEATURE_COLS,))
        cur.execute("SELECT clustering.kmeans_py(3, 100, 42)")
        print("[~] Calentamiento completado.")
    except Exception as e:
        print(f"[!] Warning en calentamiento: {e}")

    total_runs = len(SYNTHETIC_SIZES) * len(K_VALUES)
    run_n = 0

    for size in SYNTHETIC_SIZES:
        table_name = f"wine_synth_{size}"
        print(f"\n{'─'*60}")
        print(f"[+] Cargando tabla '{table_name}' ({size:,} registros)...")

        # Carga y preprocesamiento (fuera del loop de K, solo se hace una vez por tamaño)
        try:
            t_load0 = time.perf_counter()
            cur.execute(
                "SELECT clustering.load_table_py(%s::TEXT, %s::TEXT[])",
                (table_name, FEATURE_COLS)
            )
            cur.execute(
                "SELECT clustering.preprocessing_py(%s::TEXT[])",
                (FEATURE_COLS,)
            )
            t_load1 = time.perf_counter()
            load_time = t_load1 - t_load0
            print(f"[✓] Carga + preprocesamiento: {load_time:.4f} s")
        except Exception as e:
            print(f"[!] Error cargando {table_name}: {e}")
            continue

        for k in K_VALUES:
            run_n += 1
            print(f"\n  [{run_n}/{total_runs}] size={size:,} | k={k}", end=" ... ", flush=True)

            try:
                t0 = time.perf_counter()
                cur.execute(f"SELECT clustering.kmeans_py({k}, 300, 42)")
                raw = cur.fetchone()[0]
                t1 = time.perf_counter()

                payload = json.loads(raw)
                if not payload.get("ok"):
                    print(f"ERROR: {payload.get('error')}")
                    continue

                kmeans_time  = payload["training_time_seconds"]
                total_time   = t1 - t0

                print(f"kmeans={kmeans_time:.4f}s | total={total_time:.4f}s | iters={payload['iterations']}")

                results.append({
                    "tool": "Extension",
                    "records": size,
                    "k": k,
                    "attributes": len(FEATURE_COLS),
                    "kmeans_time_seconds": kmeans_time,
                    "total_time_seconds": total_time,
                    "iterations": payload["iterations"],
                    "inertia": payload["inertia"],
                })

            except Exception as e:
                print(f"ERROR: {e}")
                conn.autocommit = True  # recuperar si hay transacción rota
                continue

    cur.close()
    conn.close()

    if results:
        df = pd.DataFrame(results)
        df.to_csv(CSV_OUT, index=False)
        print(f"\n[✓] Resultados guardados en: {os.path.abspath(CSV_OUT)}")
        print(df.pivot_table(index="records", columns="k", values="kmeans_time_seconds").to_string())
    else:
        print("\n[!] No se generaron resultados.")

    return results


if __name__ == "__main__":
    run_extension_benchmark()
