# EVALUACIÓN COMPARATIVA DE RENDIMIENTO: ARQUITECTURA MEDIANAMENTE ACOPLADA PARA K-MEANS EN POSTGRESQL, WEKA Y SCIKIT-LEARN

Trabajo presentado para presentar en un cartel en el **Congreso Estatal de Ciencias Exactas y Naturales**: se mide y compara el rendimiento de **K-means** en tres configuraciones — **PostgreSQL con extensión PL/Python** (arquitectura medianamente acoplada en un SGBD), **Weka** y **Python scikit-learn** — usando el dataset **ASSISTments**.

> **Artículo de referencia**
>
> Vallejo-Cabrera, F., Timarán-Pereira, R., Chaves-Torres, A. (2025).
> *Integration of the K-means Algorithm into PostgreSQL through PL/Python Extensions:
> A Moderately Coupled Architecture.*
> Revista Facultad de Ingeniería (Rev. Fac. Ing.), Vol. 34, No. 74.
> (https://repositorio.uptc.edu.co/handle/001/21620)

---

## Propósito del proyecto

El artículo de referencia implementa K-means **dentro de PostgreSQL** y lo compara con herramientas externas concluyendo ventajas en escenarios de mayor volumen. Este repositorio **replica el diseño de la extensión** y repite el tipo de experimento con **otro dataset** y **otra tercera herramienta** (sklearn en lugar de KNIME), manteniendo parámetros de algoritmo alineados entre las tres rutas.

Objetivos concretos:

- **Comparar tiempos de forma equitativa:** mismas filas (muestra aleatoria reproducible), mismo `k-means++`, mismas `max_iter`, misma semilla del algoritmo y del muestreo.
- **Separar qué se mide:** *carga + preprocesamiento hasta poder entrenar*, *llamada completa al entrenamiento* (donde aplica) y **`tiempo_respuesta_s`** como métrica principal agregada para las tres herramientas; la extensión además expone **`tiempo_kmeans_interno_s`** (solo el `fit()` interno), comparable al “model construction time” del paper.

---

## Qué se hizo en este repositorio

| Componente | Descripción |
|------------|-------------|
| **`sql/kmeans_extension.sql`** | Instala el esquema `clustering` y las **10 funciones PL/Python** (carga, preprocesamiento, `kmeans_py`, resultados, inercia, codo, silueta, etc.), fiel al paper. |
| **`scripts/cargar_assistments_completo.py`** | Copia **`data/dataset.csv`** a PostgreSQL (tabla base del benchmark). El CSV no se versiona; instrucciones en `data/README.md`. |
| **`scripts/benchmark_test1_assistments.py`** | Prueba 1 — **Extensión**: variar tamaño de muestra (registros). Salida: `results/assistments/prueba1_extension.csv`. |
| **`scripts/benchmark_test2_assistments.py`** | Prueba 2 — **Extensión**: variar número de **atributos**. Salida: `results/assistments/prueba2_extension.csv`. |
| **`scripts/benchmark_sklearn_assistments.py`** | Mismas pruebas 1 y 2 con **scikit-learn**. CSV: `prueba1_sklearn.csv`, `prueba2_sklearn.csv`. |
| **`scripts/benchmark_weka_assistments.py`** | Mismas pruebas con **Weka** (exportación ARFF). CSV: `prueba1_weka.csv`, `prueba2_weka.csv`. |
| **`scripts/generar_graficas_assistments.py`** | Lee los CSV anteriores y genera **6 figuras** en `results/assistments/figuras/` más **`tabla_resumen_prueba1.csv`** y **`tabla_resumen_prueba2.csv`** (columna principal: `tiempo_respuesta_s`). |

### Cómo se diferencia del paper original

| Eje | Paper original | Este trabajo |
|---|---|---|
| **Dataset** | Wine Quality (21 K, contexto del paper) | **ASSISTments** (real, millones de filas en bruto; benchmarks con muestreo 1 K–1 M) |
| **Escala** | 1 K → 1 M según diseño del artículo | 1 K → 1 M con **muestreo aleatorio reproducible** sobre datos reales |
| **Tercera herramienta** | KNIME | **scikit-learn** (acoplamiento débil, sin capa Java/GUI) |
| **Métricas** | Un tiempo tipo “model construction” | Varios campos CSV; comparación principal por **`tiempo_respuesta_s`** |

Sustituir KNIME por sklearn acota la pregunta: *¿la ventaja relativa observada en el paper se explica por la integración en el SGBD o en buena medida por el overhead de la plataforma Java?*

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
   - `tiempo_kmeans_s` — **Weka / sklearn:** tiempo del proceso K-means (llamada externa). **Extensión:** tiempo de la **llamada SQL completa** a `kmeans_py` (incluye overhead del wrapper PL/Python), comparable en espíritu al tiempo medido “desde el cliente” para las otras herramientas.
   - `tiempo_kmeans_interno_s` (solo extensión) — tiempo del `fit()` dentro de `kmeans_py`; equivale al **“model construction time”** del paper.
   - `tiempo_respuesta_s` — suma carga + ejecución (métrica principal para comparar las tres herramientas en las tablas resumen del script de gráficas)

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

Valores de la columna **`tiempo_kmeans_interno_s`** en la extensión; para Weka y sklearn coincide con **`tiempo_kmeans_s`** (solo entrenamiento).

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

## Figuras generadas (`generar_graficas_assistments.py`)

Salida por defecto: **`results/assistments/figuras/`** (PNG, 150 DPI).

| Archivo | Contenido |
|---------|-----------|
| `figura1_prueba_registros.png` | Prueba 1: rendimiento variando **número de registros** (las 3 herramientas). |
| `figura2_prueba_atributos.png` | Prueba 2: rendimiento variando **número de atributos** (100 K filas). |
| `figura3_tiempo_respuesta_registros.png` | Tiempo de **respuesta total** (`tiempo_respuesta_s`) vs registros. |
| `figura4_tiempo_respuesta_atributos.png` | Tiempo de **respuesta total** vs atributos. |
| `figura5_carga_vs_ejecucion.png` | Barras apiladas **carga vs ejecución K-means** por herramienta (*k* = 5; tamaños 10 K, 100 K, 500 K, 1 M). **Los tres paneles comparten la misma escala en el eje Y** para comparación visual directa. |
| `figura6_tiempo_respuesta.png` | Comparativa global del tiempo de respuesta (paneles registros / atributos). |

Tras cada corrida del pipeline, los scripts también guardan **`tabla_resumen_prueba1.csv`** y **`tabla_resumen_prueba2.csv`** en `results/assistments/` (métrica principal: **`tiempo_respuesta_s`**). Los logs locales pueden guardarse en `results/assistments/logs/` (carpeta ignorada por git).

## Estructura del proyecto

```
fuentescartelcecen/
├── paper.md                    # Artículo de referencia (Markdown)
├── README.md
├── pyrightconfig.json
├── data/
│   ├── README.md               # Cómo obtener dataset.csv (no versionado)
│   └── dataset.csv             # (local; ignorado por git)
├── docs/
│   ├── documento_cartel.md     # Texto estructurado para el cartel CECEN
│   ├── arquitecturas.md        # Estado del arte (Timarán)
│   ├── correciones.md          # Revisión técnica de benchmarks
│   └── diagrama arquitectura medianamente acoplada.png
├── sql/
│   └── kmeans_extension.sql    # Extensión PL/Python (10 funciones, fiel al paper)
├── scripts/
│   ├── paths.py                # Rutas del repo (RESULTS, FIGURAS, DATA, SQL)
│   ├── cargar_assistments_completo.py
│   ├── benchmark_test1_assistments.py
│   ├── benchmark_test2_assistments.py
│   ├── benchmark_weka_assistments.py
│   ├── benchmark_sklearn_assistments.py
│   └── generar_graficas_assistments.py
└── results/assistments/
    ├── prueba1_extension.csv
    ├── prueba1_weka.csv
    ├── prueba1_sklearn.csv
    ├── prueba2_extension.csv
    ├── prueba2_weka.csv
    ├── prueba2_sklearn.csv
    ├── tabla_resumen_prueba1.csv
    ├── tabla_resumen_prueba2.csv
    ├── logs/                   # opcional; ignorado por git
    └── figuras/
        └── figura*.png
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

## Cómo usar el proyecto

1. **Clona el repositorio** y abre una terminal en la **raíz del repo** (donde está `README.md`).
2. **Crea la base de datos** en PostgreSQL (ej. `assistments_clustering`), habilita `plpython3u` y revisa usuario/contraseña/host en cada script bajo `scripts/` (o usa variables de entorno como `PGPASSWORD` donde estén soportadas).
3. **Dependencias:** instala los paquetes listados en [Requisitos para reproducir](#requisitos-para-reproducir) y configura el mismo Python para **PL/Python** que indica `sql/kmeans_extension.sql` (ruta a `numpy`, `pandas`, `scikit-learn`).
4. **Dataset:** descarga ASSISTments desde Kaggle, guarda el CSV como **`data/dataset.csv`** (ver `data/README.md`).
5. **Ejecuta los pasos en orden** (instalación SQL → carga → benchmarks → gráficas):

### Pipeline completo (comandos)

> El archivo `dataset.csv` (~2,8 GB) no está en Git. Fuente: <https://www.kaggle.com/datasets/nicolaswattiez/skillbuilder-data-2009-2010>

Desde la raíz del repositorio:

```powershell
# 1. Habilitar la extensión PL/Python e instalar las funciones
psql -U postgres -d assistments_clustering -f sql/kmeans_extension.sql

# 2. Colocar dataset.csv en data/ (ver data/README.md) y cargar ~6.1M filas a PostgreSQL
python scripts/cargar_assistments_completo.py

# 3. Correr los benchmarks (las tres herramientas; puedes ejecutar solo los que necesites)
python scripts/benchmark_test1_assistments.py
python scripts/benchmark_test2_assistments.py
python scripts/benchmark_sklearn_assistments.py
python scripts/benchmark_weka_assistments.py

# 4. Generar las 6 figuras y las tablas resumen (usa los CSV en results/assistments/)
python scripts/generar_graficas_assistments.py
```

**Salidas útiles:** CSV en `results/assistments/`; PNG en `results/assistments/figuras/`. Opcionalmente puedes redirigir logs a `results/assistments/logs/` (carpeta ignorada por `.gitignore`).

**Weka:** por defecto los scripts asumen instalación en `C:\Program Files\Weka-3-8-7\`; ajústalo en `benchmark_weka_assistments.py` si tu ruta difiere.

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

## Referencias principales

```
Vallejo-Cabrera, F., Timarán-Pereira, R., Chaves-Torres, A. (2025).
Integration of the K-means Algorithm into PostgreSQL through PL/Python Extensions:
A Moderately Coupled Architecture.
Revista Facultad de Ingeniería (Rev. Fac. Ing.), Vol. 34, No. 74.
DOI: 10.19053/01211129.v34.n74.2025.20737
```

Timarán Pereira, R. (2011). Arquitecturas de integración del proceso de descubrimiento de conocimiento con sistemas de gestión de bases de datos: un estado del arte. *Ingeniería y Competitividad*, 3(2), 45–55. DOI: [10.25100/iyc.v3i2.2327](https://doi.org/10.25100/iyc.v3i2.2327)
