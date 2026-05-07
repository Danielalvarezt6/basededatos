# K-Means en PostgreSQL vía PL/Python — Benchmark con Dataset ASSISTments

Repositorio del proyecto para el cartel académico presentado en el CECEN.  
Replica y extiende la metodología del artículo:

> *"Integration of the K-means Algorithm into PostgreSQL through PL/Python Extensions:  
> A Moderately Coupled Architecture"* — Revista Facultad de Ingeniería.

---

## Descripción

Se implementa y compara el algoritmo K-Means en tres herramientas:

| Herramienta | Arquitectura | Descripción |
|---|---|---|
| **Extensión PL/Python** | Acoplamiento medio | K-Means ejecutado dentro de PostgreSQL mediante funciones PL/Python |
| **Weka (SimpleKMeans)** | Acoplamiento débil | Herramienta externa; datos exportados desde PostgreSQL a ARFF |
| **Python sklearn** | Acoplamiento débil | KMeans de scikit-learn; datos descargados desde PostgreSQL vía TCP |

**Dataset:** [ASSISTments 2009–2010](https://sites.google.com/site/assistmentsdata/) — 6.1 millones de interacciones de aprendizaje con 11 variables numéricas originales.

---

## Estructura del proyecto

```
fuentescartelcecen/
│
├── prueba_fuerte.py                    # Prueba de acoplamiento fuerte (SQL puro)
├── requirements.txt                    # Dependencias Python
├── .gitignore
│
└── dataset_ASSISTment/
    │
    ├── sql/
    │   ├── kmeans_extension.sql        # Extensión PL/Python (9 funciones K-Means)
    │   └── run_kmeans_plpython_efficient.sql
    │
    ├── # ── Carga de datos ──────────────────────────────────────
    ├── cargar_assistments_completo.py  # Carga las 6.1M filas a PostgreSQL (via COPY)
    ├── cargar_wine_quality.py          # Carga Wine Quality + datos sintéticos
    ├── cargar_datos.py                 # Carga inicial ASSISTments (3 columnas)
    │
    ├── # ── Benchmarks — ASSISTments (pipeline principal) ───────
    ├── benchmark_test1_assistments.py  # Prueba 1: variando registros (1K → 1M)
    ├── benchmark_test2_assistments.py  # Prueba 2: variando atributos (3 → 11)
    ├── benchmark_weka_assistments.py   # Weka: Prueba 1 y 2
    ├── benchmark_sklearn_assistments.py# sklearn: Prueba 1 y 2
    ├── generar_graficas_assistments.py # Genera 6 figuras + tablas en español
    │
    ├── # ── Benchmarks — Wine Quality (replicación del paper) ───
    ├── benchmark_test1.py
    ├── benchmark_test2.py
    ├── benchmark_weka.py
    ├── generar_graficas.py
    │
    ├── # ── Scripts exploratorios ───────────────────────────────
    ├── prueba_debil.py                 # K-Means externo a PostgreSQL
    ├── prueba_medianamente.py          # K-Means via PL/Python
    ├── agregar_assistments.py          # Agrega ASSISTments por estudiante
    │
    └── results/
        ├── assistments/
        │   ├── prueba1_extension.csv
        │   ├── prueba1_weka.csv
        │   ├── prueba1_sklearn.csv
        │   ├── prueba2_extension.csv
        │   ├── prueba2_weka.csv
        │   ├── prueba2_sklearn.csv
        │   ├── tabla_resumen_prueba1.csv
        │   ├── tabla_resumen_prueba2.csv
        │   └── figuras/               # 6 figuras comparativas en español
        └── wine_quality/
            └── figuras/               # Figuras replicando el paper original
```

---

## Requisitos

### Software
- **PostgreSQL 18** con extensión `plpython3u` habilitada
- **Python 3.11+**
- **Weka 3.8.7** (para el benchmark de Weka)
- **Java** (incluido con Weka)

### Paquetes Python
```bash
pip install -r requirements.txt
```

### Paquetes para PL/Python (dentro de PostgreSQL)
Instalar en una carpeta accesible por el servicio de PostgreSQL:
```bash
pip install --target "C:\python_packages" numpy pandas scikit-learn
```
Y agregar al inicio de cada función PL/Python:
```python
import sys
sys.path.insert(0, r'C:\python_packages')
```

---

## Ejecución del pipeline (ASSISTments)

> **Nota:** El archivo `dataset.csv` (2.8 GB) no está incluido en el repositorio.  
> Descárgalo desde: https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment

```bash
# 1. Instalar la extensión PL/Python en PostgreSQL
#    (ejecutar en psql sobre la BD assistments_clustering)
#    CREATE EXTENSION IF NOT EXISTS plpython3u;
#    \i dataset_ASSISTment/sql/kmeans_extension.sql

# 2. Cargar el dataset completo (6.1M filas) a PostgreSQL
python dataset_ASSISTment/cargar_assistments_completo.py

# 3. Benchmarks
python dataset_ASSISTment/benchmark_test1_assistments.py
python dataset_ASSISTment/benchmark_test2_assistments.py
python dataset_ASSISTment/benchmark_weka_assistments.py
python dataset_ASSISTment/benchmark_sklearn_assistments.py

# 4. Generar gráficas y tablas en español
python dataset_ASSISTment/generar_graficas_assistments.py
```

---

## Resultados principales (k = 5 grupos)

### Tiempo de ejecución K-Means — Prueba 1

| Registros | Extensión PL/Python | Weka | Python sklearn |
|---|---|---|---|
| 1K | 0.14 s | 0.27 s | 0.01 s |
| 10K | 0.07 s | 0.32 s | 0.01 s |
| 100K | 0.55 s | 0.91 s | 0.03 s |
| 500K | 2.75 s | 3.64 s | 0.15 s |
| **1M** | **5.45 s** | **7.85 s** | **0.41 s** |

### Tiempo de respuesta total (carga + K-Means) — Prueba 1

| Registros | Extensión PL/Python | Weka | Python sklearn |
|---|---|---|---|
| 10K | 0.84 s | 2.59 s | 2.23 s |
| 100K | 8.55 s | 29.42 s | 22.83 s |
| 500K | 43.51 s | 38.10 s | 26.55 s |
| **1M** | **100.85 s** | **71.58 s** | **31.10 s** |

---

## Variables del dataset ASSISTments

| # | Variable | Descripción |
|---|---|---|
| 1 | `ms_first_response` | Tiempo de primera respuesta (ms) |
| 2 | `hint_count` | Número de pistas solicitadas |
| 3 | `attempt_count` | Número de intentos |
| 4 | `correct` | Respuesta correcta (0/1) |
| 5 | `original` | Problema original (0/1) |
| 6 | `bottom_hint` | Solicitó pista final (0/1) |
| 7 | `overlap_time` | Tiempo de solapamiento |
| 8 | `Average_confidence(FRUSTRATED)` | Confianza: frustración |
| 9 | `Average_confidence(CONFUSED)` | Confianza: confusión |
| 10 | `Average_confidence(CONCENTRATING)` | Confianza: concentración |
| 11 | `Average_confidence(BORED)` | Confianza: aburrimiento |

---

## Referencia del artículo

```
Marín-Raventós, G., & Rojas-Matarrita, A. (2023).
Integration of the K-means Algorithm into PostgreSQL through PL/Python Extensions:
A Moderately Coupled Architecture.
Revista Facultad de Ingeniería, Universidad de Antioquia.
```
