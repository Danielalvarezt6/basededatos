# Revisión Técnica de Benchmarks
**Extensión K-Means en PostgreSQL vs. Weka vs. scikit-learn**  
Dataset ASSISTments — 6.1M interacciones reales

---

## 1. Resumen Ejecutivo

Los scripts implementan una comparativa entre tres herramientas de clustering K-Means: la extensión PL/Python para PostgreSQL, Weka y scikit-learn. La revisión identifica **cuatro problemas críticos que afectan la validez científica** de los resultados, más varios problemas de importancia media y menor.

| # | Categoría | Problema | Impacto |
|---|-----------|----------|---------|
| P1 | Comparabilidad | Semilla de muestreo inconsistente entre herramientas (`0.000000042` vs `0.42`) | Las tres herramientas no procesan los mismos datos |
| P2 | Comparabilidad | Semilla de K-Means inconsistente: sklearn=42, extensión=42, Weka=`-S 10` | Los clusters no son comparables entre herramientas |
| P3 | Métricas | `tiempo_total_s` en Weka copia `tiempo_kmeans_s`, omitiendo `t_carga` | Ventaja artificial para Weka en gráficas de barras |
| P4 | Métricas | `tiempo_kmeans_s` mide cosas distintas en cada herramienta | Columna no equivalente entre las tres herramientas |
| P5 | Comparabilidad | Weka escribe un archivo ARFF a disco; extensión y sklearn operan en memoria | La comparación mezcla arquitecturas radicalmente distintas sin documentarlo |
| P6 | Metodología | El paper usa datos sintéticos de Wine Quality; los scripts usan ASSISTments reales | Divergencia entre el paper original y la implementación a validar |
| P7 | Robustez | Sin repeticiones (`n_runs=1`) — resultados sujetos a variabilidad de hardware | Conclusiones no reproducibles estadísticamente |
| P8 | Código | `generar_graficas.py` usa `col_tiempo='tiempo_total_s'` para Weka, que es incorrecto | Las tablas resumen muestran valores erróneos para Weka |
| P9 | Código | `result_py()` en la extensión lee de `clustering.cl_data` en vez de `cl_data_pre` | Los resultados muestran valores sin normalizar |
| P10 | Código | `elbow_py` y `silhouette_py` usan `n_init=10`, pero `kmeans_py` usa `n_init=1` | El modelo de evaluación es más robusto que el de producción |

---

## 2. Problemas Críticos

### P1 — Semilla de muestreo inconsistente entre herramientas

**Archivos afectados:** `benchmark_weka_assistments.py`, `benchmark_sklearn_assistments.py`, `benchmark_test1_assistments.py`

Las tres herramientas usan `crear_tabla_muestra()` pero con valores de semilla distintos:

- `benchmark_weka_assistments.py` y `benchmark_sklearn_assistments.py`:
  ```python
  cur.execute(f"SELECT setseed({semilla / 10**9:.6f})")
  # semilla=42  →  setseed(0.000000042)
  ```
- `benchmark_test1_assistments.py` (extensión):
  ```python
  cur.execute(f"SELECT setseed(0.42)")
  ```

PostgreSQL interpreta `setseed(0.000000042)` y `setseed(0.42)` como semillas completamente distintas. Cada herramienta recibe un subconjunto diferente de filas. En un dataset con distribuciones no uniformes como ASSISTments, esto produce diferencias de desempeño no atribuibles a la herramienta sino a los datos.

**Corrección:**
```python
# Usar este valor en los tres scripts:
cur.execute("SELECT setseed(0.42)")
```

---

### P2 — Semilla del algoritmo K-Means inconsistente

**Archivos afectados:** `benchmark_weka_assistments.py` (usa `-S 10`) vs. los demás (usan `seed=42`)

La semilla determina la inicialización de centroides. Con semillas distintas, los algoritmos exploran espacios diferentes del paisaje de optimización y pueden converger a soluciones locales distintas con tiempos de convergencia distintos. Parte de la diferencia de tiempos que se observa en las gráficas se debe a la semilla, no a la arquitectura.

**Corrección en `ejecutar_weka_kmeans()`:**
```python
cmd = [
    java_exe, "-Xmx6g",
    "-cp", WEKA_JAR_PATH,
    "weka.clusterers.SimpleKMeans",
    "-N", str(k),
    "-I", str(max_iter),
    "-init", "1",
    "-S", "42",   # ← cambiar de 10 a 42
    "-t", arff_path,
]
```

---

### P3 — `tiempo_total_s` de Weka copia `tiempo_kmeans_s` (omite `t_carga`)

**Archivo afectado:** `benchmark_weka_assistments.py`, funciones `prueba1_weka()` y `prueba2_weka()`

En el diccionario de resultados de Weka:
```python
resultados.append({
    ...
    "tiempo_kmeans_s":    round(tiempo_weka, 6),
    "tiempo_total_s":     round(tiempo_weka, 6),  # ← idéntico a tiempo_kmeans_s
    "tiempo_respuesta_s": round(t_respuesta, 6),
})
```

El tiempo de carga (`t_carga = descarga tabla temp + escritura ARFF`) no se incluye en `tiempo_total_s`. En cambio, sklearn sí lo incluye correctamente:
```python
"tiempo_total_s": round(t_norm + t_kmeans, 6),
```

El script `generar_graficas_assistments.py` usa `tiempo_total_s` para Weka en las tablas resumen, haciendo que Weka aparezca artificialmente más rápido.

**Corrección:**
```python
"tiempo_total_s": round(t_carga + tiempo_weka, 6),
```

---

### P4 — `tiempo_kmeans_s` mide cosas distintas en cada herramienta

**Archivos afectados:** `benchmark_test1_assistments.py`, `benchmark_test2_assistments.py`

Para la extensión, `tiempo_kmeans_s` es `payload["training_time_seconds"]`, que es el tiempo interno de `model.fit()` reportado desde dentro de PostgreSQL — sin el overhead de la llamada SQL (`t_total`). Para Weka y sklearn, `tiempo_kmeans_s` es el tiempo de la llamada externa completa. La columna no es equivalente entre las tres herramientas.

**Recomendación:** Para la extensión, usar `t_total` (la llamada SQL completa) como `tiempo_kmeans_s`, y guardar el tiempo interno como campo separado:
```python
resultados.append({
    ...
    "tiempo_kmeans_s":        round(t_total, 6),        # ← tiempo de la llamada SQL completa
    "tiempo_kmeans_interno_s": round(t_kmeans, 6),      # ← solo model.fit() interno
    "tiempo_respuesta_s":     round(t_carga + t_total, 6),
})
```

---

## 3. Problemas de Importancia Media

### P5 — La comparativa mezcla arquitecturas distintas sin documentarlo

Los tres sistemas tienen penalizaciones de I/O completamente diferentes:

- **Extensión:** los datos nunca salen de PostgreSQL. La carga es una consulta SQL interna.
- **scikit-learn:** los datos viajan PostgreSQL → psycopg2 → Python. Hay deserialización y normalización NumPy.
- **Weka:** los datos viajan PostgreSQL → Python → archivo `.arff` en disco → JVM Java. Hay serialización de texto + E/S de disco + arranque de JVM.

Esta diferencia arquitectónica es parte del argumento del paper. Sin embargo, las gráficas deberían mostrar las barras descompuestas (carga + kmeans) como métrica principal, en vez de un único número que mezcla costos estructuralmente distintos.

**Recomendación:** Usar siempre `tiempo_respuesta_s` como métrica principal de comparación (el único tiempo verdaderamente equivalente entre las tres herramientas).

---

### P6 — Divergencia entre el paper y la implementación adaptada

El paper usa el dataset **Wine Quality** (21,000 registros, 11 atributos fisicoquímicos sintéticos). Los scripts usan **ASSISTments** (6.1M interacciones reales, 11 atributos conductuales). Esto introduce diferencias importantes que no están documentadas en el código:

- El paper usa datos sintéticos; los scripts usan datos reales con otra distribución estadística.
- El paper fija 21,000 registros para el test 2; los scripts usan 100,000 sin justificación.
- El paper compara contra KNIME; los scripts reemplazan KNIME por scikit-learn.

**Recomendación:** Agregar una sección de docstring en cada archivo explicando explícitamente cómo difiere este benchmark del paper original y por qué.

---

### P7 — Sin repeticiones: resultados sujetos a ruido de hardware

Cada combinación (herramienta × tamaño × k) se ejecuta una sola vez. Los tiempos son sensibles a scheduling del SO, estado de caché de CPU, actividad de disco y garbage collection de la JVM. Además, solo la extensión tiene calentamiento previo — lo que la favorece artificialmente en datasets pequeños.

**Recomendación:** Ejecutar al menos 3 repeticiones por combinación y reportar la mediana:

```python
import statistics

N_REPETICIONES = 3

for k in K_VALORES:
    tiempos = []
    for rep in range(N_REPETICIONES):
        t = ejecutar_weka_kmeans(arff_path, k)  # o correr_kmeans(X, k)
        if t is not None:
            tiempos.append(t)
    if tiempos:
        tiempo_mediana = statistics.median(tiempos)
        resultados.append({
            ...,
            "tiempo_kmeans_s": round(tiempo_mediana, 6),
            "repeticiones":    N_REPETICIONES,
            "tiempos_raw":     str(tiempos),
        })
```

Y agregar calentamiento equivalente para Weka y sklearn antes del bucle principal.

---

## 4. Problemas Menores

### P8 — `generar_graficas_assistments.py` usa `col_tiempo` incorrecto para Weka

En la configuración `HERRAMIENTAS`, Weka tiene `"col_tiempo": "tiempo_total_s"`. Este campo es incorrecto por el problema P3. Las funciones `tabla_resumen()` y las figuras por panel usan `col_tiempo`, mostrando valores erróneos para Weka.

**Corrección:** Después de corregir P3, verificar que las tablas resumen usen `tiempo_respuesta_s`:
```python
col_usar = "tiempo_respuesta_s"  # forzar para comparación justa entre herramientas
```

---

### P9 — `result_py()` en la extensión lee de `cl_data` en vez de `cl_data_pre`

En `kmeans_extension.sql`, `result_py()` asigna los labels del modelo a las filas de `clustering.cl_data` (datos originales sin normalizar). Los labels corresponden a filas de `cl_data_pre` (normalizado). Si el preprocesamiento eliminó NULLs o cambió el número de filas, habrá desalineación de índices.

**Corrección en `kmeans_extension.sql`:**
```sql
-- Cambiar:
rows = plpy.execute("SELECT * FROM clustering.cl_data")
-- Por:
rows = plpy.execute("SELECT * FROM clustering.cl_data_pre")
```

---

### P10 — `elbow_py` y `silhouette_py` usan `n_init=10`, pero `kmeans_py` usa `n_init=1`

Las funciones de evaluación son más robustas que el modelo de producción. El usuario seleccionará un k óptimo basado en evaluaciones con `n_init=10`, pero el modelo que realmente se entrena usa `n_init=1` y puede dar peores resultados.

**Recomendación:** Unificar con una constante global:
```python
N_INIT = 1  # para benchmarks de velocidad; usar 10 para calidad de cluster
```

---

## 5. Lo que está bien implementado

- ✅ **Separación de la muestra del tiempo medido:** crear la tabla temporal fuera de la medición es metodológicamente correcto.
- ✅ **Inicialización k-means++:** los tres scripts usan k-means++ (`-init 1` en Weka, `init='k-means++'` en sklearn y la extensión). Correcto y comparable.
- ✅ **Descomposición de tiempos en columnas:** el diseño de columnas (`tiempo_carga_s`, `tiempo_kmeans_s`, `tiempo_respuesta_s`) es conceptualmente correcto y permite análisis granular.
- ✅ **Calentamiento de la extensión:** el warmup en `benchmark_test1_assistments.py` evita que los costos de compilación JIT de PL/Python afecten la primera medición.
- ✅ **Normalización min-max:** ambas implementaciones (sklearn con `MinMaxScaler`, extensión con NumPy manual) producen resultados equivalentes.
- ✅ **Uso de `GD` en la extensión:** almacenar matrices NumPy en el Global Dictionary es la forma correcta de mantener estado entre llamadas PL/Python sin escribir a disco.
- ✅ **Estructura de CSVs:** las columnas son consistentes entre archivos y permiten hacer merge directo en el script de gráficas.

---

## 6. Checklist de correcciones prioritarias

Aplicar en este orden antes de correr los benchmarks finales:

| Prioridad | Archivo | Cambio | Ref. |
|-----------|---------|--------|------|
| 🔴 1 | `benchmark_weka_assistments.py` | Cambiar `setseed` a `0.42` | P1 |
| 🔴 2 | `benchmark_sklearn_assistments.py` | Cambiar `setseed` a `0.42` | P1 |
| 🔴 3 | `benchmark_weka_assistments.py` | Cambiar `-S 10` a `-S 42` en Weka | P2 |
| 🔴 4 | `benchmark_weka_assistments.py` | `tiempo_total_s = round(t_carga + tiempo_weka, 6)` | P3 |
| 🟡 5 | `benchmark_test1/test2_assistments.py` | Usar `t_total` como `tiempo_kmeans_s` para la extensión | P4 |
| 🟡 6 | `kmeans_extension.sql` | `result_py()` debe leer de `cl_data_pre`, no `cl_data` | P9 |
| 🟡 7 | Todos los benchmarks | Agregar 3 repeticiones y reportar mediana | P7 |
| 🟡 8 | Todos los benchmarks | Agregar calentamiento idéntico para Weka y sklearn | P7 |
| 🔵 9 | `kmeans_extension.sql` | Unificar `n_init` entre `elbow_py`/`silhouette_py` y `kmeans_py` | P10 |
| 🔵 10 | `generar_graficas_assistments.py` | Forzar `tiempo_respuesta_s` en `tabla_resumen()` | P8 |

---

## 7. Nota sobre la comparativa con el paper original

El paper concluye que la extensión PL/Python supera a Weka y KNIME en escenarios de alta complejidad. Esta conclusión es **plausible y esperable** dado que la extensión evita el costo de serialización/red de arquitecturas débilmente acopladas, y Weka tiene un overhead considerable por arranque de JVM y escritura de archivos ARFF.

Sin embargo, los benchmarks en su estado actual no pueden validar esta conclusión con rigor porque los problemas P1–P4 introducen sesgos sistemáticos que favorecen artificialmente a la extensión. Después de aplicar las correcciones, el benchmark debería reproducir los resultados del paper cualitativamente, aunque con magnitudes distintas al usar ASSISTments en lugar de Wine Quality.

> **Advertencia:** La ventaja de la extensión es en gran parte arquitectónica — los datos ya están en la base de datos, sin transferencia. Si el caso de uso real requiere cargar los datos desde fuera de PostgreSQL primero, la ventaja se reduce significativamente. Esto debería mencionarse explícitamente en cualquier presentación de estos resultados.
