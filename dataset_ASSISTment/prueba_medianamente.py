import json
import os
import time
import pandas as pd
import psycopg2

# =====================================================
# CONFIGURACIÓN
# =====================================================
DB_NAME = "assistments"
DB_USER = "postgres"
DB_PASSWORD = "danonino32"
DB_HOST = "127.0.0.1" # Usamos IP directa para evitar problemas de IPv6
DB_PORT = "5432"

SIZES = [50000, 100000, 250000, 500000, 750000, 1000000]
CSV_PATH = "results/moderately_coupled_results.csv"

def execute_moderately_coupled_benchmarks():
    os.makedirs("results", exist_ok=True)
    results = []

    print("\n" + "=" * 70)
    print("🚀 BENCHMARKING - ARQUITECTURA MEDIANAMENTE ACOPLADA (In-Database)")
    print("=" * 70)

    try:
        # Usamos context managers para que la conexión siempre se cierre bien
        with psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
        ) as conn:
            with conn.cursor() as cursor:
                
                # --- WARM-UP (Calentamiento del motor de la Base de Datos) ---
                print("\n[~] Ejecutando iteración de calentamiento (Cold Start)...")
                try:
                    cursor.execute("SELECT run_kmeans(1000);")
                    print("[~] Calentamiento finalizado. PL/Python está listo y en memoria.")
                except Exception as e:
                    print(f"[!] Warning en el calentamiento (Revisa las librerías en Postgres): {e}")
                    conn.rollback()

                # --- EXPERIMENTOS ---
                for size in SIZES:
                    print(f"\n[+] Ejecutando prueba con {size:,} registros...")
                    
                    start_total = time.time()
                    
                    try:
                        cursor.execute("SELECT run_kmeans(%s);", (size,))
                        raw_result = cursor.fetchone()[0]
                    except Exception as e:
                        print(f"[!] Error ejecutando la función SQL para el tamaño {size}: {e}")
                        conn.rollback() # Limpiamos la transacción para poder continuar con el siguiente tamaño
                        continue
                        
                    end_total = time.time()
                    total_time = end_total - start_total

                    # --- PARSEO DEL JSON ---
                    try:
                        payload = json.loads(raw_result)
                    except json.JSONDecodeError:
                        print(f"[!] Error: La base de datos no devolvió un JSON válido. Devolvió:\n{raw_result}")
                        continue

                    if not payload.get("ok"):
                        print(f"[!] Error interno de PostgreSQL: {payload.get('error')}")
                        continue

                    # --- EXTRACCIÓN DE MÉTRICAS ---
                    t_sql = float(payload.get("seconds_sql_fetch", 0))
                    t_mat = float(payload.get("seconds_matrix_build", 0))
                    t_km = float(payload.get("seconds_kmeans", 0))
                    t_int = float(payload.get("seconds_plpython_total", 0))
                    
                    extract_like = t_sql + t_mat

                    print(f"[✓] Filas procesadas: {payload.get('n'):,} | Clusters: {payload.get('clusters_found')}")
                    print(f"[✓] Tiempos DB -> Extracción SQL: {t_sql:.4f} s | Matriz NumPy: {t_mat:.4f} s | K-Means: {t_km:.4f} s")
                    print(f"[✓] Tiempo interno puro (PL/Python): {t_int:.4f} s")
                    print(f"[✓] Tiempo TOTAL (Cliente, viaje de ida y vuelta): {total_time:.4f} s")

                    results.append({
                        "architecture": "moderately_coupled",
                        "rows": size,
                        "processed_rows": payload.get("n"),
                        "clusters_found": payload.get("clusters_found"),
                        "extraction_time_seconds": extract_like,
                        "kmeans_time_seconds": t_km,
                        "internal_total_time_seconds": t_int,
                        "total_time_seconds": total_time, # Este es el que vas a comparar con la débilmente acoplada
                        "cluster_0_count": payload.get("cluster_0_count", 0),
                        "cluster_1_count": payload.get("cluster_1_count", 0),
                        "cluster_2_count": payload.get("cluster_2_count", 0),
                        "cluster_3_count": payload.get("cluster_3_count", 0),
                    })

    except Exception as e:
        print(f"\n❌ Error crítico de conexión a la base de datos:\n{e}")
        return

    # =====================================================
    # EXPORTAR CSV Y MOSTRAR TABLA
    # =====================================================
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(CSV_PATH, index=False)
        
        print("\n" + "=" * 70)
        print("📊 RESULTADOS FINALES")
        print("=" * 70)
        # Mostramos la tabla en consola para una revisión rápida
        print(results_df[["rows", "extraction_time_seconds", "kmeans_time_seconds", "total_time_seconds"]].to_string(index=False))
        
        print(f"\n[✓] CSV guardado exitosamente en:\n    {os.path.abspath(CSV_PATH)}")
    else:
        print("\n[!] No se generaron métricas para guardar.")

if __name__ == "__main__":
    execute_moderately_coupled_benchmarks()