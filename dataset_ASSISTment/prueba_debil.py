import pandas as pd
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
import numpy as np
import time
import os
import gc  # Garbage Collector: vital para no saturar la RAM al extraer tantos datos

# =====================================================
# CONFIGURACIÓN
# =====================================================
DB_USER = "postgres"
DB_PASSWORD = "danonino32"
DB_HOST = "127.0.0.1"  # IP directa
DB_PORT = "5432"
DB_NAME = "assistments"

SIZES = [50000, 100000, 250000, 500000, 750000, 1000000]
K = 4
CSV_PATH = "results/weakly_coupled_results.csv"

def execute_weakly_coupled_benchmarks():
    os.makedirs("results", exist_ok=True)
    results = []

    print("\n" + "=" * 70)
    print("🚀 BENCHMARKING - ARQUITECTURA DÉBILMENTE ACOPLADA (Externa)")
    print("=" * 70)

    # Conexión PostgreSQL mediante SQLAlchemy
    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

    try:
        # --- WARM-UP (Calentamiento de conexión) ---
        print("\n[~] Ejecutando iteración de calentamiento (Cold Start)...")
        with engine.connect() as conn:
            pd.read_sql("SELECT ms_first_response FROM assistments LIMIT 1000", conn)
        print("[~] Calentamiento finalizado. Conexión establecida.")

        # --- EXPERIMENTOS ---
        for size in SIZES:
            print(f"\n[+] Ejecutando prueba con {size:,} registros...")

            query = f"""
            SELECT ms_first_response, hint_count, attempt_count
            FROM assistments
            WHERE ms_first_response > 0 AND ms_first_response < 600000
            LIMIT {size};
            """

            try:
                total_start = time.time()

                # =================================================
                # 1. EXTRACCIÓN DE DATOS (Red + Conversión a Pandas)
                # =================================================
                extraction_start = time.time()
                with engine.connect() as conn:
                    df = pd.read_sql(query, conn)
                extraction_end = time.time()
                extraction_time = extraction_end - extraction_start

                print(f"[✓] Datos cargados en RAM: {len(df):,} filas")
                print(f"[✓] Tiempo extracción (Equivalente a Fetch+Matrix): {extraction_time:.4f} s")

                # Misma precisión que PL/Python (float32) + Lloyd clásico
                X = df.to_numpy(dtype=np.float32, copy=False)

                # =================================================
                # 2. K-MEANS (alineado a run_kmeans_plpython_efficient.sql)
                # =================================================
                kmeans_start = time.time()

                kmeans = KMeans(
                    n_clusters=K,
                    init="k-means++",
                    n_init=1,  # type: ignore[arg-type]
                    max_iter=100,
                    tol=1e-4,
                    random_state=42,
                    algorithm="lloyd",
                )
                clusters = kmeans.fit_predict(X)
                
                kmeans_end = time.time()
                kmeans_time = kmeans_end - kmeans_start

                print(f"[✓] Tiempo K-Means (CPU Local): {kmeans_time:.4f} s")

                # =================================================
                # 3. TIEMPO TOTAL
                # =================================================
                total_end = time.time()
                total_time = total_end - total_start

                print(f"[✓] Tiempo TOTAL (Ida y vuelta): {total_time:.4f} s")

                # =================================================
                # 4. LIMPIEZA Y GUARDADO (Estructura idéntica al script In-Database)
                # =================================================
                unique_clusters = len(set(clusters))
                cluster_distribution = {int(c): int((clusters == c).sum()) for c in set(clusters)}

                results.append({
                    "architecture": "weakly_coupled",
                    "rows": size,
                    "processed_rows": len(df),
                    "clusters_found": unique_clusters,
                    "extraction_time_seconds": extraction_time,
                    "kmeans_time_seconds": kmeans_time,
                    "internal_total_time_seconds": 0.0,  # <-- Se agrega en 0 para que empate con el otro CSV
                    "total_time_seconds": total_time,
                    "cluster_0_count": cluster_distribution.get(0, 0),
                    "cluster_1_count": cluster_distribution.get(1, 0),
                    "cluster_2_count": cluster_distribution.get(2, 0),
                    "cluster_3_count": cluster_distribution.get(3, 0)
                })

                # Liberar RAM agresivamente
                del df
                del kmeans
                del clusters
                gc.collect()

            except Exception as e:
                print(f"[!] Error durante la iteración de tamaño {size}: {e}")
                continue

    except Exception as e:
        print(f"\n❌ Error crítico de conexión:\n{e}")

    finally:
        engine.dispose()

    # =====================================================
    # EXPORTAR CSV Y MOSTRAR TABLA
    # =====================================================
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(CSV_PATH, index=False)

        print("\n" + "=" * 70)
        print("📊 RESULTADOS FINALES")
        print("=" * 70)
        # Imprimimos las mismas columnas clave para revisión
        print(results_df[["rows", "extraction_time_seconds", "kmeans_time_seconds", "total_time_seconds"]].to_string(index=False))

        print(f"\n[✓] CSV guardado exitosamente en:\n    {os.path.abspath(CSV_PATH)}")
    else:
        print("\n[!] No se generaron métricas para guardar.")

if __name__ == "__main__":
    execute_weakly_coupled_benchmarks()