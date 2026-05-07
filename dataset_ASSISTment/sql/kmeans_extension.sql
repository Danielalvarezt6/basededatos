-- =============================================================================
-- EXTENSIÃ“N K-MEANS PARA POSTGRESQL (Arquitectura Medianamente Acoplada)
-- Replica: Vallejo-Cabrera et al., Rev. Fac. Ing., Vol. 34, No. 74 (2025)
-- DOI: 10.19053/01211129.v34.n74.2025.20737
--
-- InstalaciÃ³n:
--   psql -U postgres -d wine_quality -f kmeans_extension.sql
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS clustering;

-- =============================================================================
-- 1. load_table_py
-- =============================================================================
CREATE OR REPLACE FUNCTION clustering.load_table_py(
    source_table TEXT,
    columns      TEXT[] DEFAULT NULL
)
RETURNS TEXT
LANGUAGE plpython3u
AS $func$
import sys, json
_usp = r'C:\python_packages'
if _usp not in sys.path:
    sys.path.insert(0, _usp)

col_clause = ", ".join(f'"{c}"' for c in columns) if columns else "*"
query = f"SELECT {col_clause} FROM {source_table}"

rows = plpy.execute(query)
if not rows:
    return json.dumps({"ok": False, "error": "La tabla origen esta vacia"})

col_names = list(rows[0].keys())
data = [list(r.values()) for r in rows]

plpy.execute("DROP TABLE IF EXISTS clustering.cl_data")
col_defs = ", ".join(f'"{c}" DOUBLE PRECISION' for c in col_names)
plpy.execute(f"CREATE TABLE clustering.cl_data ({col_defs})")

plan = plpy.prepare(
    f"INSERT INTO clustering.cl_data VALUES ({', '.join(['$' + str(i+1) for i in range(len(col_names))])})",
    ["DOUBLE PRECISION"] * len(col_names)
)
for row in data:
    plpy.execute(plan, [float(v) if v is not None else None for v in row])

return json.dumps({
    "ok": True,
    "rows": len(data),
    "columns": col_names,
    "message": f"Datos cargados en clustering.cl_data desde '{source_table}'"
})
$func$;


-- =============================================================================
-- 2. load_file_py
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
-- 3. preprocessing_py
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

rows = plpy.execute("SELECT * FROM clustering.cl_data")
if not rows:
    return json.dumps({"ok": False, "error": "clustering.cl_data esta vacia"})

col_names = list(rows[0].keys())
X = np.array([[r[c] for c in col_names] for r in rows], dtype=np.float64)

num_cols = list(numeric_cols) if numeric_cols else col_names
cat_cols = list(categorical_cols) if categorical_cols else []
num_cols = [c for c in num_cols if c not in cat_cols]

# NormalizaciÃ³n Min-Max manual con numpy
col_idx = {c: i for i, c in enumerate(col_names)}
num_idx = [col_idx[c] for c in num_cols]
X_num = X[:, num_idx]
mins = X_num.min(axis=0)
maxs = X_num.max(axis=0)
rngs = np.where(maxs - mins == 0, 1.0, maxs - mins)
X[:, num_idx] = (X_num - mins) / rngs

# Para el benchmark no hay columnas categÃ³ricas en Wine Quality,
# por lo que one-hot encoding no aplica aquÃ­
final_cols = col_names

plpy.execute("DROP TABLE IF EXISTS clustering.cl_data_pre")
col_defs = ", ".join(f'"{c}" DOUBLE PRECISION' for c in final_cols)
plpy.execute(f"CREATE TABLE clustering.cl_data_pre ({col_defs})")

plan = plpy.prepare(
    f"INSERT INTO clustering.cl_data_pre VALUES ({', '.join(['$' + str(i+1) for i in range(len(final_cols))])})",
    ["DOUBLE PRECISION"] * len(final_cols)
)
for row in X.tolist():
    plpy.execute(plan, row)

return json.dumps({
    "ok": True,
    "rows": int(X.shape[0]),
    "original_cols": len(col_names),
    "processed_cols": len(final_cols),
    "numeric_normalized": num_cols,
    "categorical_encoded": cat_cols,
    "message": "Preprocesamiento completado -> clustering.cl_data_pre"
})
$func$;


-- =============================================================================
-- 4. kmeans_py
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
import sys, json, pickle, time
_usp = r'C:\python_packages'
if _usp not in sys.path:
    sys.path.insert(0, _usp)
import numpy as np
from sklearn.cluster import KMeans

rows = plpy.execute("SELECT * FROM clustering.cl_data_pre")
if not rows:
    return json.dumps({"ok": False, "error": "clustering.cl_data_pre esta vacia. Ejecuta preprocessing_py primero."})

col_names = list(rows[0].keys())
X = np.array([[r[c] for c in col_names] for r in rows], dtype=np.float64)

t0 = time.perf_counter()
model = KMeans(
    n_clusters=k,
    init="k-means++",
    n_init=10,
    max_iter=max_iter,
    random_state=seed
)
model.fit(X)
elapsed = time.perf_counter() - t0

import os
os.makedirs(r'C:\Temp', exist_ok=True)
with open(model_path, "wb") as f:
    pickle.dump({"model": model, "columns": col_names}, f)

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
-- 5. result_py
-- =============================================================================
CREATE OR REPLACE FUNCTION clustering.result_py()
RETURNS TEXT
LANGUAGE plpython3u
AS $func$
import json

if "kmeans_labels" not in GD:
    return json.dumps({"ok": False, "error": "No hay modelo en memoria. Ejecuta kmeans_py primero."})

labels    = GD["kmeans_labels"]
col_names = GD["kmeans_cols"]

rows = plpy.execute("SELECT * FROM clustering.cl_data")
if not rows:
    return json.dumps({"ok": False, "error": "clustering.cl_data esta vacia"})

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
-- 6. centroids_py
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
-- 7. summary_py
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
-- 8. inertia_py
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
-- 9. elbow_py
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
    model = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=max_iter, random_state=seed)
    model.fit(X)
    inertia = float(model.inertia_)
    plpy.execute(f"INSERT INTO clustering.cl_elbow VALUES ({k}, {inertia})")
    results.append({"k": k, "inertia": inertia})

return json.dumps({"ok": True, "k_range": [k_min, k_max], "results": results,
                   "message": "Resultados del metodo del codo en clustering.cl_elbow"})
$func$;


-- =============================================================================
-- 10. silhouette_py
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
    model  = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=max_iter, random_state=seed)
    labels = model.fit_predict(X)
    score  = float(silhouette_score(X, labels, sample_size=min(5000, len(X)), random_state=seed))
    plpy.execute(f"INSERT INTO clustering.cl_silhouette VALUES ({k}, {score})")
    results.append({"k": k, "silhouette_avg": score})

return json.dumps({"ok": True, "k_range": [k_min, k_max], "results": results,
                   "message": "Coeficientes de silueta en clustering.cl_silhouette"})
$func$;

