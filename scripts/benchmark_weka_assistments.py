"""
Benchmark Weka — Dataset ASSISTments 
Ejecuta SimpleKMeans desde línea de comandos para:
  - Prueba 1: variando número de registros (1K → 1M)
  - Prueba 2: variando número de atributos (100K fijas)

Salidas:
  results/assistments/prueba1_weka.csv
  results/assistments/prueba2_weka.csv
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import psycopg2

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
from paths import RESULTS_ASSISTMENTS

CSV_PRUEBA1_WEKA = str(RESULTS_ASSISTMENTS / "prueba1_weka.csv")
CSV_PRUEBA2_WEKA = str(RESULTS_ASSISTMENTS / "prueba2_weka.csv")

# =====================================================
# CONFIGURACIÓN
# =====================================================
WEKA_JAR_PATH = r"C:\Program Files\Weka-3-8-7\weka.jar"
JAVA_EXE_PATH = r"C:\Program Files\Weka-3-8-7\jre\jre-25.0.2-full\bin\java.exe"

DB_NAME     = "assistments_clustering"
DB_USER     = "postgres"
DB_PASSWORD = os.environ.get("PGPASSWORD", "password")
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



def df_a_arff(df: pd.DataFrame, nombre_relacion: str = "assistments") -> str:
    lines = [f"@RELATION {nombre_relacion}", ""]
    for col in df.columns:
        lines.append(f"@ATTRIBUTE {col} NUMERIC")
    lines += ["", "@DATA"]
    for _, row in df.iterrows():
        lines.append(",".join(str(round(float(v), 6)) for v in row))
    return "\n".join(lines)


def ejecutar_weka_kmeans(arff_path: str, k: int,
                         max_iter: int = 300, seed: int = 42) -> float | None:
    if not os.path.exists(WEKA_JAR_PATH):
        raise FileNotFoundError(f"No se encontró weka.jar en: {WEKA_JAR_PATH}")
    java_exe = JAVA_EXE_PATH if os.path.exists(JAVA_EXE_PATH) else "java"
    cmd = [
        java_exe, "-Xmx6g",
        "-cp", WEKA_JAR_PATH,
        "weka.clusterers.SimpleKMeans",
        "-N", str(k),
        "-I", str(max_iter),
        "-init", "1",   # 1 = k-means++ initialization (igual que extensión y sklearn)
        "-S", str(seed),
        "-t", arff_path,
    ]
    t0 = time.perf_counter()
    resultado = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    tiempo = time.perf_counter() - t0
    if resultado.returncode != 0:
        print(f"    [!] Error Weka (k={k}): {resultado.stderr[:200]}")
        return None
    return tiempo


# =====================================================
# PRUEBA 1 — Variando registros
# =====================================================
def prueba1_weka():
    os.makedirs(RESULTS_ASSISTMENTS, exist_ok=True)
    resultados = []

    print("\n" + "=" * 70)
    print("PRUEBA 1 — Weka: variando registros)")
    print("Dataset: ASSISTments")
    print("=" * 70)

    if not os.path.exists(WEKA_JAR_PATH):
        print(f"[!] weka.jar no encontrado en: {WEKA_JAR_PATH}")
        return []

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
            # Crear tabla temporal (NO se cronometra)
            nombre_tmp = crear_tabla_muestra(cur, conn, tamano, TODAS_LAS_COLUMNAS)
            print(f"[✓] Tabla temporal lista: {nombre_tmp}")
        except Exception as e:
            print(f"[!] Error preparando muestra: {e}")
            continue

        # === MEDICIÓN: leer desde tabla temp (misma condición que extensión) ===
        try:
            t0_desc = time.perf_counter()
            df = pd.read_sql(f'SELECT * FROM {nombre_tmp}', conn)
            t_descarga = time.perf_counter() - t0_desc
        except Exception as e:
            print(f"[!] Error leyendo muestra: {e}")
            continue

        # Tiempo de escritura ARFF (transferencia a disco para Weka)
        t0_arff = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=".arff", mode="w",
                                         delete=False, encoding="utf-8") as f:
            arff_path = f.name
            f.write(df_a_arff(df, f"assistments_{tamano}"))
        t_arff = time.perf_counter() - t0_arff
        t_carga = t_descarga + t_arff   # descarga tabla temp + escritura ARFF
        print(f"[✓] Descarga tabla temp: {t_descarga:.4f}s | ARFF a disco: {t_arff:.4f}s | Carga total: {t_carga:.4f}s")

        for k in K_VALORES:
            n_ejecucion += 1
            print(f"\n  [{n_ejecucion}/{total_ejecuciones}] registros={tamano:,} | k={k}",
                  end=" ... ", flush=True)
            tiempo_weka = ejecutar_weka_kmeans(arff_path, k)
            if tiempo_weka is None:
                continue
            t_respuesta = t_carga + tiempo_weka
            print(f"weka={tiempo_weka:.4f}s | carga={t_carga:.4f}s | respuesta={t_respuesta:.4f}s")
            resultados.append({
                "herramienta":         "Weka",
                "registros":           tamano,
                "num_grupos":          k,
                "num_atributos":       len(TODAS_LAS_COLUMNAS),
                "tiempo_carga_s":      round(t_carga, 6),
                "tiempo_kmeans_s":     round(tiempo_weka, 6),
                "tiempo_total_s":      round(t_carga + tiempo_weka, 6),
                "tiempo_respuesta_s":  round(t_respuesta, 6),
            })

        os.unlink(arff_path)

    if resultados:
        pd.DataFrame(resultados).to_csv(
            CSV_PRUEBA1_WEKA, index=False, encoding="utf-8")
        print(f"\n[✓] Guardado: {CSV_PRUEBA1_WEKA}")

    cur.close()
    conn.close()
    return resultados


# =====================================================
# PRUEBA 2 — Variando atributos
# =====================================================
def prueba2_weka():
    os.makedirs(RESULTS_ASSISTMENTS, exist_ok=True)
    resultados = []

    print("\n" + "=" * 70)
    print(f"PRUEBA 2 — Weka: variando atributos | {N_PRUEBA2:,} filas reales")
    print("=" * 70)

    if not os.path.exists(WEKA_JAR_PATH):
        print(f"[!] weka.jar no encontrado en: {WEKA_JAR_PATH}")
        return []

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
    t0_desc_p2 = time.perf_counter()
    df_completo = pd.read_sql(f'SELECT * FROM {nombre_tmp}', conn)
    t_descarga_p2 = time.perf_counter() - t0_desc_p2
    print(f"[✓] Descarga tabla temp: {t_descarga_p2:.4f} s")

    total_ejecuciones = len(SUBCONJUNTOS_ATRIBUTOS) * len(K_VALORES)
    n_ejecucion = 0

    for n_attrs, cols in SUBCONJUNTOS_ATRIBUTOS.items():
        print(f"\n{'─' * 60}")
        df_sub = pd.DataFrame(df_completo[cols])

        t0_arff = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=".arff", mode="w",
                                         delete=False, encoding="utf-8") as f:
            arff_path = f.name
            f.write(df_a_arff(df_sub, f"assistments_attrs{n_attrs}"))
        t_arff = time.perf_counter() - t0_arff
        t_carga = t_descarga_p2 + t_arff
        print(f"[+] {n_attrs} atributos | Descarga: {t_descarga_p2:.4f}s | ARFF: {t_arff:.4f}s | Carga total: {t_carga:.4f}s")

        for k in K_VALORES:
            n_ejecucion += 1
            print(f"\n  [{n_ejecucion}/{total_ejecuciones}] atributos={n_attrs} | k={k}",
                  end=" ... ", flush=True)
            tiempo_weka = ejecutar_weka_kmeans(arff_path, k)
            if tiempo_weka is None:
                continue
            t_respuesta = t_carga + tiempo_weka
            print(f"weka={tiempo_weka:.4f}s | carga={t_carga:.4f}s | respuesta={t_respuesta:.4f}s")
            resultados.append({
                "herramienta":         "Weka",
                "registros":           N_PRUEBA2,
                "num_grupos":          k,
                "num_atributos":       n_attrs,
                "tiempo_carga_s":      round(t_carga, 6),
                "tiempo_kmeans_s":     round(tiempo_weka, 6),
                "tiempo_total_s":      round(t_carga + tiempo_weka, 6),
                "tiempo_respuesta_s":  round(t_respuesta, 6),
            })

        os.unlink(arff_path)

    if resultados:
        pd.DataFrame(resultados).to_csv(
            CSV_PRUEBA2_WEKA, index=False, encoding="utf-8")
        print(f"\n[✓] Guardado: {CSV_PRUEBA2_WEKA}")

    cur.close()
    conn.close()
    return resultados


if __name__ == "__main__":
    prueba1_weka()
    prueba2_weka()
