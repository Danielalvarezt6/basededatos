# K-Means en PostgreSQL vía PL/Python — Réplica con dataset distinto

Repositorio del proyecto para el cartel académico presentado en el CECEN.

> **Artículo de referencia**
>
> Vallejo-Cabrera, F., Timarán-Pereira, R., Chaves-Torres, A. (2025).
> *Integration of the K-means Algorithm into PostgreSQL through PL/Python Extensions:
> A Moderately Coupled Architecture.*
> Revista Facultad de Ingeniería (Rev. Fac. Ing.), Vol. 34, No. 74.
> DOI: [10.19053/01211129.v34.n74.2025.20737](https://doi.org/10.19053/01211129.v34.n74.2025.20737)

---

## ¿Qué hace este proyecto?

El paper original propone una **extensión PL/Python para PostgreSQL** que ejecuta K-Means dentro de la base de datos (arquitectura medianamente acoplada) y la compara contra **Weka** y **KNIME** sobre el dataset **Wine Quality (21,000 registros)**. Concluye que la extensión supera a las herramientas externas en grandes volúmenes y alta complejidad.

Este proyecto **toma esa extensión como base** (implementada fielmente según las Tablas 2, 3 y 4 del paper) y prueba si esa conclusión se sostiene en un escenario distinto.

### Aportación de este trabajo

| Eje | Paper original | Este trabajo |
|---|---|---|
| **Dataset** | Wine Quality (sintético, 21K) | **ASSISTments (real, 6.1M interacciones)** |
| **Escala** | 1K → 1M (sintetizados) | 1K → 1M (muestreo aleatorio real) |
| **Tercera herramienta** | KNIME (loosely coupled, GUI Java) | **Python sklearn (loosely coupled, sin GUI)** |
| **Métricas reportadas** | "Model construction time" (un valor) | `tiempo_carga`, `tiempo_kmeans`, `tiempo_kmeans_interno`, `tiempo_respuesta` (4 desagregadas) |

Sustituir KNIME por sklearn permite responder una pregunta más estricta:

> ¿La ventaja de la extensión vs Weka/KNIME viene de **la arquitectura medianamente acoplada** o solo del **overhead de plataforma de las herramientas Java** comparadas?

Si la extensión también gana a sklearn (que es básicamente el mismo motor sin overhead), la ventaja es arquitectónica. Si pierde, la ventaja del paper se debía principalmente al overhead de Weka/KNIME.

---

## Implementación de la extensión (fiel al paper)

Las 10 funciones PL/Python siguen exactamente el diseño descrito en las Tablas 2, 3 y 4 del paper:

| Función | Tabla del paper | Descripción | Tabla auxiliar |
|---|---|---|---|
| `load_table_py(source_table)` | Tabla 2 | Lee la tabla origen y la copia al entorno de trabajo | `clustering.cl_data` |
| `load_file_py(file_path)` | Tabla 2 | Carga datos desde un CSV externo | `clustering.cl_data` |
| `preprocessing_py()` | Tabla 2 | Normaliza con MinMaxScaler de sklearn (binariza categóricas si las hay) | `clustering.cl_data_pre` |
| `kmeans_py(k, max_iter, seed)` | Tabla 3 | Ejecuta K-Means con sklearn sobre `cl_data_pre`; serializa modelo | `pickle` en disco |
| `result_py()` | Tabla 3 | Asigna cluster a cada fila | `clustering.cl_result` |
| `centroids_py()` | Tabla 3 | Exporta los centroides finales | `clustering.cl_centroids` |
| `summary_py()` | Tabla 3 | Cuenta y porcentaje por cluster | `clustering.cl_summary` |
| `inertia_py()` | Tabla 4 | WCSS y número de iteraciones | (devuelve directo) |
| `elbow_py(k_min, k_max)` | Tabla 4 | Inertia para varios k (método del codo) | `clustering.cl_elbow` |
| `silhouette_py(k_min, k_max)` | Tabla 4 | Coeficiente de silueta para varios k | `clustering.cl_silhouette` |

Parámetros de K-Means estandarizados entre las 3 herramientas para una comparación justa:
- `init = "k-means++"`
- `n_init = 1`
- `max_iter = 300`
- `random_state = 42` (semilla idéntica)

---

## Metodología de la comparación (justa)

Para que las tres herramientas compitan en igualdad de condiciones:

1. **Mismo conjunto de filas:** se crea una tabla temporal con `setseed(0.42)` + `ORDER BY random() LIMIT N` **antes** de iniciar el cronómetro. Las tres herramientas leen de esa tabla pre-creada.
2. **Mismo algoritmo:** las tres usan k-means++ con `n_init=1`, `max_iter=300`, `random_state=42`.
3. **Métricas desagregadas:**
   - `tiempo_carga_s` — desde la tabla temporal hasta dejar los datos listos para entrenar
   - `tiempo_kmeans_s` — solo el `fit()` o equivalente externo
   - `tiempo_kmeans_interno_s` (solo extensión) — `kmeans_py` aislado de su llamada SQL, comparable al "model construction time" del paper
   - `tiempo_respuesta_s` — el total que ve el usuario

### Dataset

**ASSISTments 2012-2013** — 6.1 millones de interacciones reales de aprendizaje.
Fuente: [Kaggle skillbuilder-data-2009-2010](https://www.kaggle.com/datasets/nicolaswattiez/skillbuilder-data-2009-2010).

11 atributos numéricos: `ms_first_response`, `hint_count`, `attempt_count`, `correct`, `original`, `bottom_hint`, `overlap_time`, `Average_confidence(FRUSTRATED)`, `Average_confidence(CONFUSED)`, `Average_confidence(CONCENTRATING)`, `Average_confidence(BORED)`.

---

## Resultados principales (k = 5 grupos)

### Resultado 1 — Tiempo de respuesta TOTAL al usuario (carga + preprocesamiento + kmeans)

| Registros | Extensión PL/Python | Weka | Python sklearn |
|---|---|---|---|
| 1K | 0.14 s | 0.52 s | 0.22 s |
| 10K | 0.91 s | 0.89 s | 0.05 s |
| 21K | 1.68 s | 1.73 s | 0.11 s |
| 100K | 7.47 s | 7.00 s | 0.59 s |
| 500K | 39.13 s | 19.02 s | 3.00 s |
| **1M** | **79.05 s** | **31.27 s** | **6.78 s** |

### Resultado 2 — Tiempo del K-Means aislado ("model construction time" del paper)

| Registros | Extensión PL/Python | Weka | Python sklearn |
|---|---|---|---|
| 1K | 0.026 s | 0.35 s | 0.009 s |
| 10K | 0.018 s | 0.57 s | 0.010 s |
| 21K | 0.028 s | 0.72 s | 0.019 s |
| 100K | 0.13 s | 0.97 s | 0.05 s |
| 500K | 0.56 s | 3.19 s | 0.30 s |
| **1M** | **1.01 s** | **5.39 s** | **0.58 s** |

### Lectura de los resultados

1. **El paper SÍ tiene razón en la parte algorítmica.** Si solo se mide el K-Means aislado (Resultado 2), la extensión es **5× más rápida que Weka** en 1M filas, replicando la conclusión del paper.

2. **Pero el tiempo de respuesta total cuenta otra historia.** Cuando se mide el flujo completo (Resultado 1), la extensión paga un overhead caro al materializar las tablas auxiliares `cl_data` y `cl_data_pre` (escritura de 1M filas dos veces) — y termina **2.5× más lenta que Weka** en 1M filas.

3. **sklearn gana siempre.** Sin overhead de plataforma (sin JVM como Weka, sin escritura a tablas SQL como la extensión), procesa el mismo K-Means **6× más rápido que la extensión** en 1M filas. Esto sugiere que la ventaja reportada por el paper vs KNIME se debía en buena parte al overhead de Java de KNIME, no exclusivamente a la arquitectura.

4. **Conclusión académica:** la arquitectura medianamente acoplada de la extensión es genuinamente competitiva en el algoritmo, pero su diseño basado en tablas auxiliares persistentes (Tabla 2 del paper) introduce un overhead que la hace inviable para análisis ad-hoc sobre grandes volúmenes. Sería interesante para trabajo futuro evaluar variantes que eviten la materialización intermedia.

---

## Estructura del proyecto

```
fuentescartelcecen/
├── paper.md                                 # Artículo de referencia
├── README.md
├── correciones.md                           # Bitácora técnica de las correcciones aplicadas
│
└── dataset_ASSISTment/
    │
    ├── sql/
    │   └── kmeans_extension.sql             # Extensión PL/Python (10 funciones, fiel al paper)
    │
    ├── # ── Carga de datos ─────────────────
    ├── cargar_assistments_completo.py       # Carga 6.1M filas a PostgreSQL (COPY)
    │
    ├── # ── Benchmarks (pipeline principal) ─
    ├── benchmark_test1_assistments.py       # Extensión: variando registros (1K → 1M)
    ├── benchmark_test2_assistments.py       # Extensión: variando atributos (3 → 11)
    ├── benchmark_weka_assistments.py        # Weka: Prueba 1 y Prueba 2
    ├── benchmark_sklearn_assistments.py     # sklearn: Prueba 1 y Prueba 2
    ├── generar_graficas_assistments.py      # Genera figuras comparativas
    │
    └── results/assistments/
        ├── prueba1_extension.csv            # Tiempos extensión variando registros
        ├── prueba1_weka.csv
        ├── prueba1_sklearn.csv
        ├── prueba2_extension.csv            # Tiempos variando atributos
        ├── prueba2_weka.csv
        ├── prueba2_sklearn.csv
        ├── tabla_resumen_prueba1.csv
        ├── tabla_resumen_prueba2.csv
        └── figuras/                         # PNG comparativos
```

---

## Especificaciones del entorno de pruebas

Todos los benchmarks reportados se ejecutaron en el siguiente entorno:

### Hardware

| Componente | Especificación |
|---|---|
| **CPU** | Intel Core i7-13700HX (13ª gen.) — 16 núcleos / 24 hilos, 2.10 GHz base |
| **Caché** | L2: 14 MB · L3: 30 MB |
| **Memoria RAM** | 16 GB DDR5 @ 4800 MHz (SK Hynix) |
| **Almacenamiento** | SSD NVMe de 1 TB (Hynix HFS001TEJ9X125N) |
| **GPU integrada** | Intel UHD Graphics |
| **GPU dedicada** | NVIDIA GeForce RTX 4050 Laptop GPU (4 GB) — *no utilizada en los benchmarks* |

### Software

| Componente | Versión |
|---|---|
| **Sistema operativo** | Microsoft Windows 11 Home (build 26200, 64 bits) |
| **PostgreSQL** | 18.3 (con extensión `plpython3u` habilitada) |
| **Python (intérprete del cliente y de PL/Python)** | 3.13.13 |
| **Weka** | 3.8.7 |
| **Java (Weka)** | OpenJDK 25.0.2 LTS (incluido con Weka) |

### Bibliotecas Python

| Paquete | Versión |
|---|---|
| `numpy` | 2.4.4 |
| `pandas` | 3.0.2 |
| `scikit-learn` | 1.8.0 |
| `matplotlib` | 3.10.9 |
| `psycopg2` | 2.9.12 |

> **Nota sobre paralelismo:** todos los algoritmos se ejecutaron en condiciones equivalentes con `n_init=1`. scikit-learn y la extensión PL/Python pueden aprovechar múltiples hilos a través de los backends de NumPy/BLAS, mientras que Weka (`SimpleKMeans`) corre en un solo hilo. Esta diferencia es estructural de cada herramienta y no se modificó para preservar la comparación con configuraciones por defecto.

---

## Requisitos para reproducir

### Software
- **PostgreSQL 18** con `plpython3u` habilitado (`CREATE EXTENSION plpython3u;`)
- **Python 3.13** (o 3.11+)
- **Weka 3.8.7** instalado en `C:\Program Files\Weka-3-8-7\`

### Paquetes Python (cliente de los benchmarks)

```bash
pip install pandas numpy scikit-learn matplotlib psycopg2-binary
```

### Paquetes Python (para PL/Python dentro de PostgreSQL)

```bash
pip install --target "C:\python_packages" numpy pandas scikit-learn
```

(Cada función PL/Python ya incluye `sys.path.insert(0, r'C:\python_packages')`.)

---

## Ejecución del pipeline completo

> El archivo `dataset.csv` (2.8 GB) no está versionado. Descárgalo desde: <https://www.kaggle.com/datasets/nicolaswattiez/skillbuilder-data-2009-2010>

```powershell
# 1. Habilitar la extensión PL/Python e instalar las funciones
psql -U postgres -d assistments_clustering -f dataset_ASSISTment/sql/kmeans_extension.sql

# 2. Cargar el dataset completo (6.1M filas) a PostgreSQL
python dataset_ASSISTment/cargar_assistments_completo.py

# 3. Correr los benchmarks (las cuatro herramientas)
python dataset_ASSISTment/benchmark_test1_assistments.py
python dataset_ASSISTment/benchmark_test2_assistments.py
python dataset_ASSISTment/benchmark_sklearn_assistments.py
python dataset_ASSISTment/benchmark_weka_assistments.py

# 4. Generar todas las figuras y tablas comparativas
python dataset_ASSISTment/generar_graficas_assistments.py
```

---

## Parámetros de comparación

Estandarizados entre las tres herramientas para que la comparación sea justa:

| Parámetro | Valor | Aplicado a |
|---|---|---|
| Inicialización | `k-means++` | Extensión, sklearn, Weka (`-init 1`) |
| `n_init` | `1` | Las 3 |
| `max_iter` | `300` | Las 3 |
| Semilla del algoritmo | `42` | Las 3 (Weka: `-S 42`) |
| Semilla del muestreo | `setseed(0.42)` | Las 3 leen la **misma** tabla temporal |
| Valores de k | 2 → 10 | Las 3 |
| Tamaños (Prueba 1) | 1K, 2K, 5K, 10K, 21K, 50K, 100K, 500K, 1M | Las 3 |
| Atributos (Prueba 2) | 3, 5, 7, 9, 11 (con 100K filas fijas) | Las 3 |

---

## Referencia

```
Vallejo-Cabrera, F., Timarán-Pereira, R., Chaves-Torres, A. (2025).
Integration of the K-means Algorithm into PostgreSQL through PL/Python Extensions:
A Moderately Coupled Architecture.
Revista Facultad de Ingeniería (Rev. Fac. Ing.), Vol. 34, No. 74.
DOI: 10.19053/01211129.v34.n74.2025.20737
```
