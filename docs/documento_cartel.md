# Texto estructurado para el cartel (CECEN)

Documento de apoyo con secciones: título, introducción, materiales y métodos, resultados, conclusiones y referencias.  
Para la versión completa en un solo archivo, mantén sincronizado este documento con el **README** principal (tablas de resultados y rutas del repo).

---

## 1. Título

**EVALUACIÓN COMPARATIVA DE RENDIMIENTO: ARQUITECTURA MEDIANAMENTE ACOPLADA PARA K-MEANS EN POSTGRESQL, WEKA Y SCIKIT-LEARN**

---

## 2. Introducción

El clustering particional **K-means** es ampliamente usado en minería de datos descriptiva. Integrarlo en el **SGBD** responde a la necesidad de analizar grandes volúmenes sin exportar datos a herramientas externas. La literatura clasifica la integración como **débil**, **mediana** o **fuerte** acoplamiento (**Timarán**, estado del arte).

**Vallejo-Cabrera, Timarán-Pereira y Chaves-Torres (2025)** implementan K-means en **PostgreSQL** vía **PL/Python** (acoplamiento **mediano**) y lo comparan con **Weka** y **KNIME** sobre **Wine Quality**. Este proyecto **replica el diseño experimental** pero usa **ASSISTments** (datos reales, ~6,1 M filas) y sustituye KNIME por **scikit-learn**, para separar el efecto de la arquitectura del efecto del **stack Java** de las herramientas originales.

---

## 3. Materiales y métodos

- **Marco:** taxonomía de arquitecturas DCBD–SGBD (`docs/arquitecturas.md`).
- **Extensión:** diez funciones PL/Python según tablas 2–4 del paper; tablas `clustering.cl_data` y `clustering.cl_data_pre`; `kmeans_py` con scikit-learn; parámetros alineados: k-means++, `n_init=1`, `max_iter=300`, semilla 42.
- **Benchmarks:** **Weka** (SimpleKMeans, ARFF) y **Python + sklearn** (débilmente acoplados).
- **Equidad:** misma muestra (`setseed(0.42)`, tabla temporal previa), mismos *k* y tamaños (Prueba 1: 1 K–1 M registros; Prueba 2: 100 K filas, 3–11 atributos).
- **Métricas:** `tiempo_carga_s`, `tiempo_kmeans_s`, `tiempo_kmeans_interno_s` (extensión), `tiempo_respuesta_s`.
- **Entorno:** ver tabla «Especificaciones» en el README del repositorio.

---

## 4. Resultados y discusión

*(Insertar aquí las tablas y figuras del README: tiempo total vs tiempo K-means aislado para k=5.)*

Síntesis: la extensión **supera a Weka en el tiempo del K-means puro** a gran escala (coherente con el paper); el **tiempo total** puede favorecer a Weka o sklearn por el costo de **materializar tablas** intermedias y por ausencia de overhead JVM en sklearn.

---

## 5. Conclusiones

1. Réplica fiel del diseño del paper + evaluación en **ASSISTments** con **tres herramientas** y metodología corregida (semillas, métricas Weka, comparabilidad).
2. Ventaja **algorítmica** de la extensión vs Weka **reproducida**; **sklearn** más rápido en global al evitar persistencia SQL intermedia y JVM.
3. El diseño basado en tablas del paper es **auditable** pero **costoso en E/S** a 1 M filas; trabajo futuro: reducir materialización manteniendo trazabilidad.

---

## 6. Referencias

1. Vallejo-Cabrera, F.-M., Timarán-Pereira, R., Chaves-Torres, A. (2025). *Integration of the K-means Algorithm into PostgreSQL through PL/Python Extensions: A Moderately Coupled Architecture.* Rev. Fac. Ing., 34(74). DOI: 10.19053/01211129.v34.n74.2025.20737  
2. Timarán Pereira, R. (2011). *Arquitecturas de integración del proceso de descubrimiento de conocimiento con sistemas de gestión de bases de datos: un estado del arte.* Ingeniería y Competitividad, 3(2), 45–55. DOI: 10.25100/iyc.v3i2.2327  
