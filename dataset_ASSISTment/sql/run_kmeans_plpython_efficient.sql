-- K-means medianamente acoplado: PL/Python eficiente
-- 1) Una sola consulta devuelve tres array_agg (sin bucle Python fila a fila)
-- 2) K-means solo con NumPy (sin sklearn/joblib dentro del servidor)
-- 3) Hiperparámetros alineados con prueba_debil.py: k=4, k-means++, n_init=1,
--    max_iter=100, tol=1e-4, random_state=42, matriz float32
--
-- Instalación (ajusta tipos si tu Postgres usa otro nombre de función):
--   psql -U postgres -d assistments -f run_kmeans_plpython_efficient.sql
--
-- Si existía run_kmeans(integer) con otro cuerpo, CREATE OR REPLACE lo sustituye
-- si la firma coincide (mismo nombre y tipos de argumentos).

CREATE OR REPLACE FUNCTION run_kmeans(sample_size integer)
RETURNS text
LANGUAGE plpython3u
AS $func$
import json
import time
import numpy as np

_K = 4
_RNG_SEED = 42
_MAX_ITER = 100
_TOL = 1e-4


def _kmeans_plus_plus(X, k, rng):
    """Inicialización k-means++ (misma familia que sklearn por defecto)."""
    n, _d = X.shape
    centers = np.empty((k, X.shape[1]), dtype=X.dtype)
    idx0 = int(rng.integers(n))
    centers[0] = X[idx0]
    closest_sq = np.sum((X - centers[0]) ** 2, axis=1)
    for j in range(1, k):
        s = closest_sq.sum()
        if s <= 0:
            centers[j] = X[int(rng.integers(n))]
            continue
        probs = closest_sq / s
        idx = int(rng.choice(n, p=probs))
        centers[j] = X[idx]
        d_new = np.sum((X - centers[j]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, d_new)
    return centers


def _lloyd(X, k, centers, rng, max_iter, tol):
    n, d = X.shape
    centers = np.asarray(centers, dtype=np.float32)
    for _ in range(max_iter):
        dists_sq = np.sum(
            (X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2
        )
        labels = np.argmin(dists_sq, axis=1)

        new_centers = np.zeros((k, d), dtype=np.float32)
        np.add.at(new_centers, labels, X)
        counts = np.bincount(labels, minlength=k).astype(np.float32)
        for j in range(k):
            if counts[j] == 0:
                new_centers[j] = X[int(rng.integers(n))]
                counts[j] = 1.0
        new_centers /= counts[:, np.newaxis]

        shift = float(np.linalg.norm(new_centers - centers))
        centers = new_centers
        if shift < tol:
            break

    dists_sq = np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)
    labels = np.argmin(dists_sq, axis=1)
    return labels, centers


if sample_size is None or sample_size < 1:
    return json.dumps({"ok": False, "error": "sample_size invalid"})

t_wall0 = time.perf_counter()

# Una fila, tres vectores densos — evita dict por fila en Python
plan = plpy.prepare(
    """
    WITH subset AS (
        SELECT ms_first_response, hint_count, attempt_count
        FROM assistments
        WHERE ms_first_response > 0
          AND ms_first_response < 600000
        LIMIT $1
    ),
    numbered AS (
        SELECT
            row_number() OVER (
                ORDER BY ms_first_response, hint_count, attempt_count, ctid
            ) AS rn,
            ms_first_response,
            hint_count,
            attempt_count
        FROM subset
    )
    SELECT
        array_agg(ms_first_response ORDER BY rn) AS c0,
        array_agg(hint_count ORDER BY rn) AS c1,
        array_agg(attempt_count ORDER BY rn) AS c2
    FROM numbered
    """,
    ["integer"],
)

t_before_fetch = time.perf_counter()
rows = plpy.execute(plan, [int(sample_size)])
t_after_fetch = time.perf_counter()

if not rows:
    return json.dumps({"ok": False, "error": "no rows from query"})

row = rows[0]
c0, c1, c2 = row["c0"], row["c1"], row["c2"]
if c0 is None or len(c0) == 0:
    return json.dumps({"ok": False, "error": "empty sample"})

t_before_stack = time.perf_counter()
X = np.column_stack(
    (
        np.asarray(c0, dtype=np.float32),
        np.asarray(c1, dtype=np.float32),
        np.asarray(c2, dtype=np.float32),
    )
)
t_after_stack = time.perf_counter()

rng = np.random.default_rng(_RNG_SEED)

t_km0 = time.perf_counter()
centers = _kmeans_plus_plus(X, _K, rng)
labels, _final = _lloyd(X, _K, centers, rng, _MAX_ITER, _TOL)
t_km1 = time.perf_counter()

t_wall1 = time.perf_counter()

cluster_counts = [int(np.sum(labels == j)) for j in range(_K)]

out = {
    "ok": True,
    "n": int(X.shape[0]),
    "clusters_found": _K,
    "seconds_sql_fetch": t_after_fetch - t_before_fetch,
    "seconds_matrix_build": t_after_stack - t_before_stack,
    "seconds_kmeans": t_km1 - t_km0,
    "seconds_plpython_total": t_wall1 - t_wall0,
    "cluster_0_count": cluster_counts[0],
    "cluster_1_count": cluster_counts[1],
    "cluster_2_count": cluster_counts[2],
    "cluster_3_count": cluster_counts[3],
}
return json.dumps(out)
$func$;
