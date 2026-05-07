import codecs
import os
import traceback

import pandas as pd
import psycopg
from psycopg import sql
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


def detectar_encoding_csv(ruta: str) -> str:
    """Elige una codificación razonable según BOM y una muestra del archivo."""
    with open(ruta, "rb") as f:
        muestra = f.read(min(262_144, os.path.getsize(ruta)))

    if muestra.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if muestra.startswith(codecs.BOM_UTF16_LE):
        return "utf-16-le"
    if muestra.startswith(codecs.BOM_UTF16_BE):
        return "utf-16-be"

    for enc in ("utf-8", "cp1252"):
        try:
            muestra.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


# --- PostgreSQL (psycopg v3: evita UnicodeDecodeError típico de psycopg2 en Windows) ---
PG_HOST = "localhost"
PG_PORT = 5432
PG_USER = "postgres"
PG_PASSWORD = "danonino32"
PG_DBNAME = "assistments"


def asegurar_base_datos() -> None:
    """Si la base PG_DBNAME no existe, la crea conectando a la base «postgres»."""
    with psycopg.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname="postgres",
        user=PG_USER,
        password=PG_PASSWORD,
        autocommit=True,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (PG_DBNAME,),
            )
            if cur.fetchone() is None:
                cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(PG_DBNAME)))
                print(f"✅ Base de datos «{PG_DBNAME}» creada (antes no existía).")


engine = create_engine(
    URL.create(
        "postgresql+psycopg",
        username=PG_USER,
        password=PG_PASSWORD,
        host=PG_HOST,
        port=PG_PORT,
        database=PG_DBNAME,
    ),
    pool_pre_ping=True,
)

# PON AQUÍ LA RUTA A TU ARCHIVO CSV
ruta_csv = r"C:\Users\deat_\Downloads\fuentescartelcecen\dataset_ASSISTment\dataset.csv"

# Solo columnas numéricas que K-Means necesita
columnas_utiles = ["ms_first_response", "hint_count", "attempt_count"]

print("🚀 Iniciando carga masiva de datos...")
encoding_csv = detectar_encoding_csv(ruta_csv)
print(f"📄 CSV detectado como: {encoding_csv}")

chunksize = 100_000
contador = 0

try:
    asegurar_base_datos()

    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
        conn.commit()

    if encoding_csv.startswith("utf"):
        lector = pd.read_csv(
            ruta_csv,
            usecols=columnas_utiles,
            chunksize=chunksize,
            low_memory=False,
            encoding=encoding_csv,
            encoding_errors="replace",
        )
    else:
        lector = pd.read_csv(
            ruta_csv,
            usecols=columnas_utiles,
            chunksize=chunksize,
            low_memory=False,
            encoding=encoding_csv,
        )

    for chunk in lector:
        chunk = chunk.dropna()
        chunk = chunk[chunk["ms_first_response"] > 0]

        chunk.to_sql(name="assistments", con=engine, if_exists="append", index=False)

        contador += len(chunk)
        print(f"✅ Se han subido {contador:,.0f} filas exitosamente...")

    print("\n🎉 ¡CARGA COMPLETADA! Los datos están en PostgreSQL.")

except Exception as e:
    print(f"\n❌ Error: {e}")
    traceback.print_exc()
    print(
        "\nComprueba que el servicio PostgreSQL esté arrancado y que el usuario pueda crear bases.\n"
        "Creación manual:\n"
        f'  psql -U postgres -c "CREATE DATABASE {PG_DBNAME};"'
    )
