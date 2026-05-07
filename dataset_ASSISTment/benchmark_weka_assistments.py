"""
Benchmark Weka — Dataset ASSISTments (6.1M interacciones reales)
Ejecuta SimpleKMeans desde línea de comandos para:
  - Prueba 1: variando número de registros (1K → 1M, todos reales)
  - Prueba 2: variando número de atributos (100K filas reales fijas)

Salidas:
  results/assistments/prueba1_weka.csv
  results/assistments/prueba2_weka.csv
"""

import os
import subprocess
import tempfile
import time
import pandas as pd
import psycopg2

# =====================================================
# CONFIGURACIÓN
# =====================================================
WEKA_JAR_PATH = r"C:\Program Files\Weka-3-8-7\weka.jar"
JAVA_EXE_PATH = r"C:\Program Files\Weka-3-8-7\jre\jre-25.0.2-full\bin\java.exe"

DB_NAME     = "assistments_clustering"
DB_USER     = "postgres"
DB_PASSWORD = "danonino32"
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
def obtener_muestra_df(tamano: int, cols: list, semilla: int = 42) -> pd.DataFrame:
    """Descarga una muestra aleatoria real de PostgreSQL como DataFrame."""
    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
        host=DB_HOST, port=DB_PORT
    )
    col_clause = ", ".join(f'"{c}"' for c in cols)
    query = f"""
        SELECT setseed({semilla / 10**9:.6f});
        SELECT {col_clause}
        FROM {TABLA_REAL}
        ORDER BY random()
        LIMIT {tamano}
    """
    # Ejecutar las dos sentencias por separado
    with conn.cursor() as cur:
        cur.execute(f"SELECT setseed({semilla / 10**9:.6f})")
        conn.commit()
    df = pd.read_sql(
        f"SELECT {col_clause} FROM {TABLA_REAL} ORDER BY random() LIMIT {tamano}",
        conn
    )
    conn.close()
    return df


def df_a_arff(df: pd.DataFrame, nombre_relacion: str = "assistments") -> str:
    lines = [f"@RELATION {nombre_relacion}", ""]
    for col in df.columns:
        lines.append(f"@ATTRIBUTE {col} NUMERIC")
    lines += ["", "@DATA"]
    for _, row in df.iterrows():
        lines.append(",".join(str(round(float(v), 6)) for v in row))
    return "\n".join(lines)


def ejecutar_weka_kmeans(arff_path: str, k: int,
                         max_iter: int = 300, seed: int = 10) -> float | None:
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
    os.makedirs("results/assistments", exist_ok=True)
    resultados = []

    print("\n" + "=" * 70)
    print("PRUEBA 1 — Weka: variando registros (datos REALES)")
    print("Dataset: ASSISTments — 6.1M interacciones reales")
    print("=" * 70)

    if not os.path.exists(WEKA_JAR_PATH):
        print(f"[!] weka.jar no encontrado en: {WEKA_JAR_PATH}")
        return []

    total_ejecuciones = len(TAMANOS) * len(K_VALORES)
    n_ejecucion = 0

    for tamano in TAMANOS:
        print(f"\n{'─' * 60}")
        print(f"[+] Descargando muestra real de {tamano:,} filas...")
        try:
            t0_desc = time.perf_counter()
            df = obtener_muestra_df(tamano, TODAS_LAS_COLUMNAS)
            t_descarga = time.perf_counter() - t0_desc
            print(f"[✓] Descarga BD: {t_descarga:.4f} s")
        except Exception as e:
            print(f"[!] Error: {e}")
            continue

        # Tiempo de escritura ARFF (transferencia a disco para Weka)
        t0_arff = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=".arff", mode="w",
                                         delete=False, encoding="utf-8") as f:
            arff_path = f.name
            f.write(df_a_arff(df, f"assistments_{tamano}"))
        t_arff = time.perf_counter() - t0_arff
        t_carga = t_descarga + t_arff   # descarga BD + escritura disco
        print(f"[✓] Descarga BD: {t_descarga:.4f}s | ARFF a disco: {t_arff:.4f}s | Carga total: {t_carga:.4f}s")

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
                "tiempo_total_s":      round(tiempo_weka, 6),
                "tiempo_respuesta_s":  round(t_respuesta, 6),
            })

        os.unlink(arff_path)

    if resultados:
        pd.DataFrame(resultados).to_csv(
            "results/assistments/prueba1_weka.csv", index=False, encoding="utf-8")
        print("\n[✓] Guardado: results/assistments/prueba1_weka.csv")
    return resultados


# =====================================================
# PRUEBA 2 — Variando atributos
# =====================================================
def prueba2_weka():
    os.makedirs("results/assistments", exist_ok=True)
    resultados = []

    print("\n" + "=" * 70)
    print(f"PRUEBA 2 — Weka: variando atributos | {N_PRUEBA2:,} filas reales")
    print("=" * 70)

    if not os.path.exists(WEKA_JAR_PATH):
        print(f"[!] weka.jar no encontrado en: {WEKA_JAR_PATH}")
        return []

    print(f"\n[+] Descargando muestra fija de {N_PRUEBA2:,} filas reales...")
    try:
        t0_desc_p2 = time.perf_counter()
        df_completo = obtener_muestra_df(N_PRUEBA2, TODAS_LAS_COLUMNAS)
        t_descarga_p2 = time.perf_counter() - t0_desc_p2
        print(f"[✓] Descarga BD: {t_descarga_p2:.4f} s")
    except Exception as e:
        print(f"[!] Error: {e}")
        return []

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
                "tiempo_total_s":      round(tiempo_weka, 6),
                "tiempo_respuesta_s":  round(t_respuesta, 6),
            })

        os.unlink(arff_path)

    if resultados:
        pd.DataFrame(resultados).to_csv(
            "results/assistments/prueba2_weka.csv", index=False, encoding="utf-8")
        print("\n[✓] Guardado: results/assistments/prueba2_weka.csv")
    return resultados


if __name__ == "__main__":
    prueba1_weka()
    prueba2_weka()
