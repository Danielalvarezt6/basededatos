-- =============================================================================
-- EXTENSIÓN K-MEANS PARA POSTGRESQL — ARQUITECTURA MEDIANAMENTE ACOPLADA
-- Réplica de: Vallejo-Cabrera, Timarán-Pereira, Chaves-Torres
-- "Integration of the K-means algorithm into PostgreSQL through PL/Python
--  extensions: a moderately coupled architecture"
-- Rev. Fac. Ing., Vol. 34, No. 74 (2025).
-- DOI: 10.19053/01211129.v34.n74.2025.20737
--
-- Esta implementación sigue fielmente las funciones descritas en el paper
-- (Tablas 2, 3 y 4): cada función persiste sus resultados en una tabla
-- auxiliar dentro del esquema `clustering`, tal como especifica el paper.
--
-- Instalación:
--   psql -U postgres -d <basededatos> -f kmeans_extension.sql
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS clustering;


-- =============================================================================
-- 1. load_table_py  (Tabla 2 del paper)
-- "It loads the records from an existing PostgreSQL table into the
--  extension's working environment. It stores the data in the temporary
--  table cl_data."
-- =============================================================================
CREATE OR REPLACE FUNCTION clustering.load_table_py(
    source_table TEXT,
    columns      TEXT[] DEFAULT NULL
)
RETURNS TEXT
LANGUAGE plpython3u
AS $func$
import json

# Determinar columnas a copiar
if columns:
    col_clause = ", ".join(f'"{c}"' for c in columns)
    col_names  = list(columns)
else:
    info = plpy.execute(f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = '{source_table.split('.')[-1]}'
        ORDER BY ordinal_position
    """)
    if not info:
        return json.dumps({"ok": False, "error": f"Tabla {source_table} no encontrada"})
    col_names  = [r["column_name"] for r in info]
    col_clause = ", ".join(f'"{c}"' for c in col_names)

# Crear tabla cl_data con la misma estructura (m × n)
plpy.execute("DROP TABLE IF EXISTS clustering.cl_data")
col_defs = ", ".join(f'"{c}" DOUBLE PRECISION' for c in col_names)
plpy.execute(f"CREATE TABLE clustering.cl_data ({col_defs})")

# Transferir los registros desde la tabla origen a cl_data
plpy.execute(f"""
    INSERT INTO clustering.cl_data ({col_clause})
    SELECT {col_clause} FROM {source_table}
""")

count = plpy.execute("SELECT COUNT(*) AS n FROM clustering.cl_data")[0]["n"]
return json.dumps({
    "ok": True,
    "rows": int(count),
    "columns": col_names,
    "message": f"Datos cargados en clustering.cl_data desde '{source_table}'"
})
$func$;


-- =============================================================================
-- 2. load_file_py  (Tabla 2 del paper)
-- "It enables data ingestion from an external CSV file, transferring the
--  data into the working environment. It stores the data in cl_data."
-- =============================================================================
CREATE OR REPLACE FUNCTION clustering.load_file_py(
    file_path TEXT,
    columns   TEXT[]  DEFAULT NULL,
    delimiter TEXT    DEFAULT ','
)
RETURNS TEXT
LANGUAGE plpython3u
AS $func$
import sys, json
_usp = r'C:\python_packages'
if _usp not in sys.path:
    sys.path.insert(0, _usp)
import pandas as pd

usecols = columns if columns else None
df = pd.read_csv(file_path, usecols=usecols, sep=delimiter).dropna()
col_names = list(df.columns)

plpy.execute("DROP TABLE IF EXISTS clustering.cl_data")
col_defs = ", ".join(f'"{c}" DOUBLE PRECISION' for c in col_names)
plpy.execute(f"CREATE TABLE clustering.cl_data ({col_defs})")

plan = plpy.prepare(
    f"INSERT INTO clustering.cl_data VALUES ({', '.join(['$' + str(i+1) for i in range(len(col_names))])})",
    ["DOUBLE PRECISION"] * len(col_names)
)
for row in df.itertuples(index=False):
    plpy.execute(plan, list(row))

return json.dumps({
    "ok": True,
    "rows": len(df),
    "columns": col_names,
    "message": f"Datos cargados en clustering.cl_data desde '{file_path}'"
})
$func$;


-- =============================================================================
-- 3. preprocessing_py  (Tabla 2 del paper)
-- "Internally, it applies normalization. It stores the result in the
--  temporary table cl_data_pre."
--
-- Lee de cl_data, normaliza con MinMaxScaler de sklearn, y persiste en
-- cl_data_pre. Las variables categóricas se binarizan (one-hot) si se
-- indican; en este dataset todas las variables son numéricas.
-- =============================================================================
CREATE OR REPLACE FUNCTION clustering.preprocessing_py(
    numeric_cols     TEXT[] DEFAULT NULL,
    categorical_cols TEXT[] DEFAULT NULL
)
RETURNS TEXT
LANGUAGE plpython3u
AS $func$
import sys, json
_usp = r'C:\python_packages'
if _usp not in sys.path:
    sys.path.insert(0, _usp)
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Leer cl_data
rows = plpy.execute("SELECT * FROM clustering.cl_data")
if not rows:
    return json.dumps({"ok": False, "error": "clustering.cl_data esta vacia. Ejecuta load_table_py primero."})

col_names = list(rows[0].keys())
df = pd.DataFrame([dict(r) for r in rows], columns=col_names)

# Determinar columnas numéricas y categóricas
num_cols = list(numeric_cols) if numeric_cols else col_names
cat_cols = list(categorical_cols) if categorical_cols else []
num_cols = [c for c in num_cols if c not in cat_cols]

# Binarización de categóricas (si las hay)
if cat_cols:
    df = pd.get_dummies(df, columns=cat_cols, dtype=float)

# Normalización Min-Max de las numéricas
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols].astype(float))

final_cols = list(df.columns)
X_norm     = df.values.astype(np.float64)

# Persistir en cl_data_pre (estructura m × p, p ≥ n por la binarización)
plpy.execute("DROP TABLE IF EXISTS clustering.cl_data_pre")
col_defs = ", ".join(f'"{c}" DOUBLE PRECISION' for c in final_cols)
plpy.execute(f"CREATE TABLE clustering.cl_data_pre ({col_defs})")

# Inserción por lotes (1000 filas por INSERT) para 1M filas
BATCH = 1000
n_rows, n_cols = X_norm.shape
for start in range(0, n_rows, BATCH):
    chunk = X_norm[start:start + BATCH]
    values = ", ".join(
        "(" + ",".join(repr(float(v)) for v in row) + ")"
        for row in chunk
    )
    plpy.execute(f"INSERT INTO clustering.cl_data_pre VALUES {values}")

return json.dumps({
    "ok": True,
    "rows": int(n_rows),
    "original_cols": len(col_names),
    "processed_cols": int(n_cols),
    "numeric_normalized": num_cols,
    "categorical_encoded": cat_cols,
    "message": "Preprocesamiento completado en clustering.cl_data_pre"
})
$func$;


-- =============================================================================
-- 4. kmeans_py  (Tabla 3 del paper)
-- "Central function that executes the k-means algorithm using the
--  preprocessed data stored in cl_data_pre, with parameters (K, number of
--  iterations, seed, etc.) defined by the user. The trained model is
--  serialized and externally persisted."
-- =============================================================================
CREATE OR REPLACE FUNCTION clustering.kmeans_py(
    k          INTEGER,
    max_iter   INTEGER DEFAULT 300,
    seed       INTEGER DEFAULT 42,
    model_path TEXT    DEFAULT 'C:\\Temp\\kmeans_model.pickle'
)
RETURNS TEXT
LANGUAGE plpython3u
AS $func$
import sys, json, pickle, time, os
_usp = r'C:\python_packages'
if _usp not in sys.path:
    sys.path.insert(0, _usp)
import numpy as np
from sklearn.cluster import KMeans

# Leer datos preprocesados desde cl_data_pre (como especifica el paper)
rows = plpy.execute("SELECT * FROM clustering.cl_data_pre")
if not rows:
    return json.dumps({"ok": False, "error": "clustering.cl_data_pre esta vacia. Ejecuta preprocessing_py primero."})

col_names = list(rows[0].keys())
X = np.array([[r[c] for c in col_names] for r in rows], dtype=np.float64)

# Ejecución k-means con los parámetros del usuario
t0 = time.perf_counter()
model = KMeans(
    n_clusters=k,
    init="k-means++",
    n_init=1,
    max_iter=max_iter,
    random_state=seed
)
model.fit(X)
elapsed = time.perf_counter() - t0

# Serialización externa del modelo (como especifica el paper)
os.makedirs(r'C:\Temp', exist_ok=True)
with open(model_path, "wb") as f:
    pickle.dump({"model": model, "columns": col_names}, f)

# Persistir labels y modelo en GD para que las demás funciones los reutilicen
GD["kmeans_labels"] = model.labels_.tolist()
GD["kmeans_model"]  = model
GD["kmeans_cols"]   = col_names

return json.dumps({
    "ok": True,
    "k": k,
    "iterations": int(model.n_iter_),
    "inertia": float(model.inertia_),
    "training_time_seconds": round(elapsed, 6),
    "model_saved": model_path,
    "message": f"K-Means ejecutado con k={k} en {elapsed:.4f}s"
})
$func$;


-- =============================================================================
-- 5. result_py  (Tabla 3 del paper)
-- "It stores the data with the cluster assignments in the table cl_result.
--  Its structure is m × (n+1), where the additional column corresponds to
--  the cluster label."
-- =============================================================================
CREATE OR REPLACE FUNCTION clustering.result_py()
RETURNS TEXT
LANGUAGE plpython3u
AS $func$
import json

if "kmeans_labels" not in GD:
    return json.dumps({"ok": False, "error": "No hay modelo en memoria. Ejecuta kmeans_py primero."})

labels = GD["kmeans_labels"]

rows = plpy.execute("SELECT * FROM clustering.cl_data_pre")
if not rows:
    return json.dumps({"ok": False, "error": "clustering.cl_data_pre esta vacia"})

col_names = list(rows[0].keys())

plpy.execute("DROP TABLE IF EXISTS clustering.cl_result")
col_defs = ", ".join(f'"{c}" DOUBLE PRECISION' for c in col_names) + ', "cluster_label" INTEGER'
plpy.execute(f"CREATE TABLE clustering.cl_result ({col_defs})")

plan = plpy.prepare(
    f"INSERT INTO clustering.cl_result VALUES ({', '.join(['$' + str(i+1) for i in range(len(col_names) + 1)])})",
    ["DOUBLE PRECISION"] * len(col_names) + ["INTEGER"]
)
for i, row in enumerate(rows):
    vals = [float(row[c]) if row[c] is not None else None for c in col_names] + [labels[i]]
    plpy.execute(plan, vals)

return json.dumps({
    "ok": True,
    "rows": len(labels),
    "message": "Resultados guardados en clustering.cl_result"
})
$func$;


-- =============================================================================
-- 6. centroids_py  (Tabla 3 del paper)
-- "It displays the final centroids of the k-means model. It stores the
--  centroids in the cl_centroids table. Its structure is k × (n+1)."
-- =============================================================================
CREATE OR REPLACE FUNCTION clustering.centroids_py()
RETURNS TEXT
LANGUAGE plpython3u
AS $func$
import json

if "kmeans_model" not in GD:
    return json.dumps({"ok": False, "error": "No hay modelo en memoria. Ejecuta kmeans_py primero."})

model     = GD["kmeans_model"]
col_names = GD["kmeans_cols"]
centers   = model.cluster_centers_

plpy.execute("DROP TABLE IF EXISTS clustering.cl_centroids")
col_defs = ", ".join(f'"{c}" DOUBLE PRECISION' for c in col_names) + ', "cluster_label" INTEGER'
plpy.execute(f"CREATE TABLE clustering.cl_centroids ({col_defs})")

plan = plpy.prepare(
    f"INSERT INTO clustering.cl_centroids VALUES ({', '.join(['$' + str(i+1) for i in range(len(col_names) + 1)])})",
    ["DOUBLE PRECISION"] * len(col_names) + ["INTEGER"]
)
for j, center in enumerate(centers):
    plpy.execute(plan, [float(v) for v in center] + [j])

return json.dumps({
    "ok": True,
    "k": int(model.n_clusters),
    "columns": col_names,
    "message": "Centroides guardados en clustering.cl_centroids"
})
$func$;


-- =============================================================================
-- 7. summary_py  (Tabla 3 del paper)
-- "It provides a statistical summary of the clustering by counting the
--  number of data points in each cluster. It stores the summary in
--  cl_summary (k × 3): cluster label, total count, percentage."
-- =============================================================================
CREATE OR REPLACE FUNCTION clustering.summary_py()
RETURNS TEXT
LANGUAGE plpython3u
AS $func$
import sys, json
_usp = r'C:\python_packages'
if _usp not in sys.path:
    sys.path.insert(0, _usp)
import numpy as np

if "kmeans_labels" not in GD:
    return json.dumps({"ok": False, "error": "No hay modelo en memoria. Ejecuta kmeans_py primero."})

labels = np.array(GD["kmeans_labels"])
total  = len(labels)
k      = GD["kmeans_model"].n_clusters

plpy.execute("DROP TABLE IF EXISTS clustering.cl_summary")
plpy.execute("CREATE TABLE clustering.cl_summary (cluster_label INTEGER, count INTEGER, percentage DOUBLE PRECISION)")

summary = []
for j in range(k):
    cnt = int(np.sum(labels == j))
    pct = round(cnt / total * 100, 4)
    plpy.execute(f"INSERT INTO clustering.cl_summary VALUES ({j}, {cnt}, {pct})")
    summary.append({"cluster": j, "count": cnt, "percentage": pct})

return json.dumps({"ok": True, "total": total, "summary": summary,
                   "message": "Resumen guardado en clustering.cl_summary"})
$func$;


-- =============================================================================
-- 8. inertia_py  (Tabla 4 del paper)
-- "It retrieves and displays the total inertia (Within-Cluster Sum of
--  Squares - WCSS) and the number of iterations used in kmeans_py."
-- =============================================================================
CREATE OR REPLACE FUNCTION clustering.inertia_py()
RETURNS TEXT
LANGUAGE plpython3u
AS $func$
import json

if "kmeans_model" not in GD:
    return json.dumps({"ok": False, "error": "No hay modelo en memoria. Ejecuta kmeans_py primero."})

model = GD["kmeans_model"]
return json.dumps({
    "ok": True,
    "inertia_wcss": float(model.inertia_),
    "iterations_used": int(model.n_iter_),
    "k": int(model.n_clusters)
})
$func$;


-- =============================================================================
-- 9. elbow_py  (Tabla 4 del paper)
-- "Repeatedly executes the k-means algorithm varying the value of K within
--  a defined range. It stores the results in cl_elbow (q × 2): K value,
--  total inertia."
-- =============================================================================
CREATE OR REPLACE FUNCTION clustering.elbow_py(
    k_min    INTEGER DEFAULT 2,
    k_max    INTEGER DEFAULT 10,
    max_iter INTEGER DEFAULT 300,
    seed     INTEGER DEFAULT 42
)
RETURNS TEXT
LANGUAGE plpython3u
AS $func$
import sys, json
_usp = r'C:\python_packages'
if _usp not in sys.path:
    sys.path.insert(0, _usp)
import numpy as np
from sklearn.cluster import KMeans

rows = plpy.execute("SELECT * FROM clustering.cl_data_pre")
if not rows:
    return json.dumps({"ok": False, "error": "clustering.cl_data_pre esta vacia"})

col_names = list(rows[0].keys())
X = np.array([[r[c] for c in col_names] for r in rows], dtype=np.float64)

plpy.execute("DROP TABLE IF EXISTS clustering.cl_elbow")
plpy.execute("CREATE TABLE clustering.cl_elbow (k INTEGER, inertia DOUBLE PRECISION)")

results = []
for k in range(k_min, k_max + 1):
    model = KMeans(n_clusters=k, init="k-means++", n_init=1, max_iter=max_iter, random_state=seed)
    model.fit(X)
    inertia = float(model.inertia_)
    plpy.execute(f"INSERT INTO clustering.cl_elbow VALUES ({k}, {inertia})")
    results.append({"k": k, "inertia": inertia})

return json.dumps({"ok": True, "k_range": [k_min, k_max], "results": results,
                   "message": "Resultados del metodo del codo en clustering.cl_elbow"})
$func$;


-- =============================================================================
-- 10. silhouette_py  (Tabla 4 del paper)
-- "It calculates the Silhouette Coefficient for various values of K. It
--  stores the coefficients in cl_silhouette (q × 2)."
-- =============================================================================
CREATE OR REPLACE FUNCTION clustering.silhouette_py(
    k_min    INTEGER DEFAULT 2,
    k_max    INTEGER DEFAULT 10,
    max_iter INTEGER DEFAULT 300,
    seed     INTEGER DEFAULT 42
)
RETURNS TEXT
LANGUAGE plpython3u
AS $func$
import sys, json
_usp = r'C:\python_packages'
if _usp not in sys.path:
    sys.path.insert(0, _usp)
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

rows = plpy.execute("SELECT * FROM clustering.cl_data_pre")
if not rows:
    return json.dumps({"ok": False, "error": "clustering.cl_data_pre esta vacia"})

col_names = list(rows[0].keys())
X = np.array([[r[c] for c in col_names] for r in rows], dtype=np.float64)

plpy.execute("DROP TABLE IF EXISTS clustering.cl_silhouette")
plpy.execute("CREATE TABLE clustering.cl_silhouette (k INTEGER, silhouette_avg DOUBLE PRECISION)")

results = []
for k in range(k_min, k_max + 1):
    model  = KMeans(n_clusters=k, init="k-means++", n_init=1, max_iter=max_iter, random_state=seed)
    labels = model.fit_predict(X)
    score  = float(silhouette_score(X, labels, sample_size=min(5000, len(X)), random_state=seed))
    plpy.execute(f"INSERT INTO clustering.cl_silhouette VALUES ({k}, {score})")
    results.append({"k": k, "silhouette_avg": score})

return json.dumps({"ok": True, "k_range": [k_min, k_max], "results": results,
                   "message": "Coeficientes de silueta en clustering.cl_silhouette"})
$func$;
