# K-Means en PostgreSQL vía PL/Python — Benchmark con Dataset ASSISTments

Repositorio del proyecto para el cartel académico presentado en el CECEN.  
Replica y extiende la metodología del artículo:

> *"Integration of the K-means Algorithm into PostgreSQL through PL/Python Extensions:  
> A Moderately Coupled Architecture"*  
> Vallejo-Cabrera, F., Timarán-Pereira, R., Chaves-Torres, A. — Revista Facultad de Ingeniería, 2025.

---

## ¿Qué hace este proyecto?

El paper original valida una extensión de PostgreSQL que ejecuta K-Means desde adentro de la base de datos (arquitectura medianamente acoplada) y la compara contra **Weka** y **KNIME** usando el dataset **Wine Quality**.

Este proyecto **replica esa comparación pero con el dataset ASSISTments** (6.1 millones de interacciones reales de estudiantes), reemplazando KNIME por **Python scikit-learn** como tercera herramienta de comparación.

### Herramientas comparadas

| Herramienta | Arquitectura | Cómo accede a los datos |
|---|---|---|
| **Extensión PL/Python** | Acoplamiento medio | K-Means corre dentro de PostgreSQL; datos leídos internamente con `load_table_py` |
| **Weka (SimpleKMeans)** | Acoplamiento débil | Descarga desde PostgreSQL vía `pd.read_sql` y escribe un archivo `.arff` a disco |
| **Python sklearn** | Acoplamiento débil | Descarga desde PostgreSQL vía `pd.read_sql` y normaliza en memoria externa |

### Dataset

**ASSISTments 2012–2013** — 6.1 millones de interacciones de aprendizaje.  
Fuente: [https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment](https://www.kaggle.com/datasets/nicolaswattiez/skillbuilder-data-2009-2010)

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

## Estructura del proyecto

```
fuentescartelcecen/
│
├── paper.md                            # Artículo de referencia (Markdown)
├── README.md
├── .gitignore
│
└── dataset_ASSISTment/
    │
    ├── sql/
    │   ├── kmeans_extension.sql            # Extensión PL/Python (9 funciones K-Means)
    │   └── run_kmeans_plpython_efficient.sql
    │
    ├── # ── Carga de datos ──────────────────────────────────────────
    ├── cargar_assistments_completo.py      # Carga 6.1M filas a PostgreSQL (COPY)
    ├── cargar_wine_quality.py              # Carga Wine Quality a PostgreSQL
    ├── cargar_datos.py                     # Carga inicial ASSISTments (3 columnas)
    ├── agregar_assistments.py              # Agrega interacciones por estudiante
    │
    ├── # ── Benchmarks — ASSISTments (pipeline principal) ───────────
    ├── benchmark_test1_assistments.py      # Extensión: variando registros (1K → 1M)
    ├── benchmark_test2_assistments.py      # Extensión: variando atributos (3 → 11)
    ├── benchmark_weka_assistments.py       # Weka: Prueba 1 y Prueba 2
    ├── benchmark_sklearn_assistments.py    # sklearn: Prueba 1 y Prueba 2
    ├── generar_graficas_assistments.py     # Genera figuras comparativas en español
    │
    ├── # ── Benchmarks — Wine Quality (replicación directa del paper) ──
    ├── benchmark_test1.py                  # Extensión vs Weka: variando registros
    ├── benchmark_test2.py                  # Extensión vs Weka: variando atributos
    ├── benchmark_weka.py                   # Weka (Wine Quality)
    ├── generar_graficas.py                 # Figuras Wine Quality
    │
    ├── # ── Scripts exploratorios ────────────────────────────────────
    ├── prueba_debil.py                     # K-Means externo a PostgreSQL
    ├── prueba_medianamente.py              # K-Means via PL/Python
    │
    └── results/
        ├── assistments/
        │   ├── prueba1_extension.csv       # Extensión: tiempos variando registros
        │   ├── prueba1_weka.csv            # Weka: tiempos variando registros
        │   ├── prueba1_sklearn.csv         # sklearn: tiempos variando registros
        │   ├── prueba2_extension.csv       # Extensión: tiempos variando atributos
        │   ├── prueba2_weka.csv            # Weka: tiempos variando atributos
        │   ├── prueba2_sklearn.csv         # sklearn: tiempos variando atributos
        │   ├── tabla_resumen_prueba1.csv
        │   ├── tabla_resumen_prueba2.csv
        │   ├── _datos_figura5.py           # Genera figura 5 con escala Y compartida
        │   └── figuras/                    # Figuras comparativas (PNG)
        └── wine_quality/
            └── figuras/                    # Figuras replicando el paper original
```

---

## Requisitos

### Software
- **PostgreSQL 18** con extensión `plpython3u` habilitada
- **Python 3.11+**
- **Weka 3.8.7** instalado en `C:\Program Files\Weka-3-8-7\`
- **Java** (incluido con Weka)

### Paquetes Python

```bash
pip install pandas numpy scikit-learn matplotlib psycopg2-binary
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
> Descárgalo desde: (https://www.kaggle.com/datasets/nicolaswattiez/skillbuilder-data-2009-2010)

```bash
# 1. Instalar la extensión en PostgreSQL
#    (ejecutar en psql sobre la BD assistments_clustering)
#    CREATE EXTENSION IF NOT EXISTS plpython3u;
#    \i dataset_ASSISTment/sql/kmeans_extension.sql

# 2. Cargar el dataset completo (6.1M filas) a PostgreSQL
python dataset_ASSISTment/cargar_assistments_completo.py

# 3. Correr los benchmarks
python dataset_ASSISTment/benchmark_test1_assistments.py
python dataset_ASSISTment/benchmark_test2_assistments.py
python dataset_ASSISTment/benchmark_weka_assistments.py
python dataset_ASSISTment/benchmark_sklearn_assistments.py

# 4. Generar figuras y tablas
python dataset_ASSISTment/generar_graficas_assistments.py

# 5. Generar figura 5 (carga vs ejecución con escala compartida)
python dataset_ASSISTment/results/assistments/_datos_figura5.py
```

---

## Resultados principales (k = 5 grupos)

### Prueba 1 — Tiempo de ejecución K-Means variando registros

| Registros | Extensión PL/Python | Weka | Python sklearn |
|---|---|---|---|
| 1K | 0.13 s | 0.25 s | 0.01 s |
| 10K | 0.08 s | 0.36 s | 0.01 s |
| 100K | 0.73 s | 0.96 s | 0.03 s |
| 500K | 3.28 s | 3.51 s | 0.15 s |
| **1M** | **6.60 s** | **8.63 s** | **0.50 s** |

### Prueba 1 — Tiempo de carga de datos

| Registros | Extensión PL/Python | Weka | Python sklearn |
|---|---|---|---|
| 10K | 0.73 s | 2.27 s | 2.22 s |
| 100K | 7.49 s | 28.51 s | 22.80 s |
| 500K | 37.35 s | 34.46 s | 26.40 s |
| **1M** | **87.24 s** | **63.73 s** | **30.69 s** |

> La extensión es más lenta en carga a gran escala porque `load_table_py` lee las filas
> de la tabla temporal una a una desde el cursor interno de PL/Python, mientras que
> Weka y sklearn usan `pd.read_sql` que transfiere los datos en batch desde PostgreSQL.

---

## Referencia

```
Vallejo-Cabrera, F., Timarán-Pereira, R., Chaves-Torres, A. (2025).
Integration of the K-means Algorithm into PostgreSQL through PL/Python Extensions:
A Moderately Coupled Architecture.
Revista Facultad de Ingeniería (Rev. Fac. Ing.), Vol. 34, No. 74.
DOI: 10.19053/01211129.v34.n74.2025.20737
```
