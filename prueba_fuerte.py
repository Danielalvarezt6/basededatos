import json
import os
import time
import pandas as pd
import psycopg2

DB_NAME = "assistments"
DB_USER = "postgres"
DB_PASSWORD = "danonino32"
DB_HOST = "127.0.0.1"
DB_PORT = "5432"

SIZES = [50000, 100000, 250000, 500000, 750000, 1000000]
CSV_PATH = "results/tightly_coupled_results.csv"

def execute_tightly_coupled_benchmarks():
    os.makedirs("results", exist_ok=True)
    results = []

    print("\n" + "=" * 70)
    print("🚀 BENCHMARKING - ARQUITECTURA FUERTEMENTE ACOPLADA (Pure SQL)")
    print("=" * 70)

    try:
        with psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT) as conn:
            with conn.cursor() as cursor:
                for size in SIZES:
                    print(f"\n[+] Ejecutando SQL K-Means con {size:,} registros (Puede tardar bastante)...")
                    
                    start_total = time.time()
                    try:
                        cursor.execute("SELECT kmeans_fuertemente_acoplado(%s, 10);", (size,))
                        raw_result = cursor.fetchone()[0]
                    except Exception as e:
                        print(f"[!] Error: {e}")
                        conn.rollback()
                        continue
                        
                    total_time = time.time() - start_total
                    payload = raw_result if isinstance(raw_result, dict) else json.loads(raw_result)

                    print(f"[✓] Filas procesadas: {payload.get('rows_processed'):,}")
                    print(f"[✓] Tiempo interno de SQL: {payload.get('total_time_seconds'):.4f} s")
                    print(f"[✓] Tiempo TOTAL (Cliente): {total_time:.4f} s")

                    results.append({
                        "architecture": "tightly_coupled",
                        "rows": size,
                        "total_time_seconds": total_time
                    })

    except Exception as e:
        print(f"\n❌ Error de conexión:\n{e}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(CSV_PATH, index=False)
        print("\n📊 RESULTADOS FINALES:\n", df.to_string(index=False))

if __name__ == "__main__":
    execute_tightly_coupled_benchmarks()