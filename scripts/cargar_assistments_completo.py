"""
Carga el dataset ASSISTments a PostgreSQL
Base de datos: assistments_clustering
Tabla destino: interacciones

Columnas originales del dataset:
  ms_first_response, hint_count, attempt_count, correct, original,
  bottom_hint, overlap_time,
  Average_confidence(FRUSTRATED), Average_confidence(CONFUSED),
  Average_confidence(CONCENTRATING), Average_confidence(BORED)

Nota: los nombres con paréntesis se mantienen exactamente como en el dataset
      (se usan con comillas en SQL).
"""

import csv
import io
import os
import sys
from pathlib import Path

import pandas as pd
import psycopg2

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
from paths import DEFAULT_DATASET_CSV

CSV_PATH    = str(DEFAULT_DATASET_CSV)
DB_USER     = "postgres"
DB_PASSWORD = os.environ.get("PGPASSWORD", "password")
DB_HOST     = "127.0.0.1"
DB_PORT     = "5432"
DB_NAME     = "assistments_clustering"
TABLA       = "interacciones"
CHUNK_SIZE  = 100_000

# Columnas del dataset ASSISTments que se cargarán
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

# Columnas de confianza que pueden tener NaN (se rellenan con 0)
CONF_COLS = [c for c in COLUMNAS if c.startswith("Average_confidence")]

# Nombres de columna en SQL (con comillas para los que tienen paréntesis)
COLS_SQL = ", ".join(
    f'"{c}"' if "(" in c else c for c in COLUMNAS
)


def conectar():
    return psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
        host=DB_HOST, port=DB_PORT
    )


def crear_tabla(conn):
    col_defs = []
    for c in COLUMNAS:
        nombre_sql = f'"{c}"' if "(" in c else c
        col_defs.append(f"    {nombre_sql} DOUBLE PRECISION")
    ddl = f"DROP TABLE IF EXISTS {TABLA};\nCREATE TABLE {TABLA} (\n" + \
          ",\n".join(col_defs) + "\n);"
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()
    print(f"[✓] Tabla '{TABLA}' creada.")


def cargar_con_copy(conn):
    total = 0
    lector = pd.read_csv(
        CSV_PATH,
        usecols=COLUMNAS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    with conn.cursor() as cur:
        for i, chunk in enumerate(lector):
            # Convertir a numérico
            for col in COLUMNAS:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

            # Filtrar filas inválidas
            chunk = chunk[chunk["ms_first_response"].notna() &
                          (chunk["ms_first_response"] > 0)]
            chunk[CONF_COLS] = chunk[CONF_COLS].fillna(0.0)
            chunk = chunk.dropna(subset=[c for c in COLUMNAS if c not in CONF_COLS])

            if chunk.empty:
                continue

            # Serializar a CSV en memoria y usar COPY para inserción rápida
            buf = io.StringIO()
            chunk[COLUMNAS].to_csv(buf, index=False, header=False,
                                   quoting=csv.QUOTE_MINIMAL)
            buf.seek(0)
            cur.copy_expert(
                f"COPY {TABLA} ({COLS_SQL}) FROM STDIN WITH (FORMAT CSV)",
                buf
            )
            conn.commit()

            total += len(chunk)
            print(f"  Chunk {i+1:3d} → {total:>9,} filas cargadas")

    return total


if __name__ == "__main__":
    print("=" * 65)
    print("  CARGA COMPLETA ASSISTments → PostgreSQL (via COPY)")
    print(f"  Base de datos: {DB_NAME}  |  Tabla: {TABLA}")
    print("=" * 65)

    conn = conectar()
    try:
        crear_tabla(conn)
        total = cargar_con_copy(conn)
        print(f"\n[✓] Carga completada: {total:,} filas en '{TABLA}'.")

        # Verificación final
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {TABLA}")
            n = cur.fetchone()[0]
        print(f"[✓] Verificación PostgreSQL: {n:,} filas en '{TABLA}'.")
    finally:
        conn.close()
