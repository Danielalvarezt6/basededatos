# [cite_start]ARQUITECTURAS DE INTEGRACIÓN DEL PROCESO DE DESCUBRIMIENTO DE CONOCIMIENTO CON SISTEMAS DE GESTIÓN DE BASES DE DATOS: UN ESTADO DEL ARTE [cite: 1]

[cite_start]**Autor:** Ricardo Timarán Pereira [cite: 2]
* [cite_start]Master of Science en Ingeniería - Universidad Politécnica de Donetsk (Ucrania). [cite: 3]
* [cite_start]Especialista en Multimedia Educativa - Universidad Antonio Nariño. [cite: 3]
* [cite_start]Candidato a Doctor en Ingeniería - Universidad del Valle. [cite: 4]
* [cite_start]Profesor Asistente del Departamento de Ingeniería de Sistemas - Universidad del Nariño. [cite: 5]

---

## [cite_start]RESUMEN [cite: 6]
* [cite_start]Las investigaciones iniciales en Descubrimiento de Conocimiento en Bases de Datos (DCBD) se centraron en definir modelos de descubrimiento de patrones y desarrollar algoritmos[cite: 7].
* [cite_start]Investigaciones posteriores se han enfocado en integrar el DCBD con sistemas de bases de datos, lo que ha generado sistemas clasificados en tres categorías de arquitectura: débilmente, medianamente y fuertemente acopladas con un Sistema de Gestión de Bases de Datos (SGBD)[cite: 8].
* [cite_start]El artículo revisa el estado del arte de estas arquitecturas como parte de una investigación doctoral enfocada en nuevas primitivas SQL para DCBD en arquitecturas fuertemente acopladas[cite: 12].
* [cite_start]**Palabras claves:** Descubrimiento de Conocimiento en Bases de Datos, Minería de Datos[cite: 13].

---

## [cite_start]1. INTRODUCCIÓN [cite: 22]
* [cite_start]El crecimiento explosivo de datos ha superado los métodos de análisis tradicionales, creando la necesidad de herramientas que transformen datos en conocimiento útil de forma automática e inteligente[cite: 23].
* [cite_start]El DCBD es un proceso iterativo e interactivo para identificar patrones válidos, novedosos y útiles[cite: 25, 26].
* [cite_start]Este proceso incluye pasos como Selección, Preprocesamiento/Limpieza, Transformación/Reducción, Minería de Datos e Interpretación/Evaluación[cite: 26].
* [cite_start]La Minería de Datos es la etapa principal donde se descubren los patrones[cite: 27].
* [cite_start]Existe una necesidad latente de integrar los sistemas de descubrimiento de conocimiento con las bases de datos, lo que resulta en enfoques de integración clasificados en arquitecturas débilmente, medianamente y fuertemente acopladas[cite: 29, 30, 31, 32, 33].

---

## [cite_start]2. ARQUITECTURA DCBD DÉBILMENTE ACOPLADA [cite: 42]
* [cite_start]**Definición:** Los algoritmos de Minería de Datos están en una capa externa al SGBD, y su integración se limita a comandos de lectura y escritura[cite: 43].
* [cite_start]**Funcionamiento:** Los procesos corren en un espacio de direccionamiento distinto al del SGBD[cite: 44]. [cite_start]El SGBD da el almacenamiento persistente, pero el procesamiento ocurre externamente[cite: 45].
* [cite_start]**Herramientas representativas:** Es la arquitectura más común[cite: 46]. [cite_start]Herramientas como Alice, C5.0_RuleQuest, Qyield y CoverStory solo soportan la minería y requieren pre y posprocesamiento[cite: 47, 48]. [cite_start]Otras como Clementine, DBMiner, Intelligent Miner y Quest combinan clasificación, visualización y consultas[cite: 58].
* [cite_start]**Implementación:** Funciona mediante SQL embebido, leyendo los datos registro por registro a través de ODBC, JDBC o cursores SQL[cite: 60, 61].
* [cite_start]**Ventajas:** Su principal ventaja es la portabilidad[cite: 62].
* [cite_start]**Desventajas:** Poca escalabilidad (cargan todos los datos en memoria) y bajo rendimiento (lectura de registros uno a uno)[cite: 63, 64]. [cite_start]Los algoritmos no pueden aprovechar el optimizador de consultas del SGBD por estar fuera del núcleo[cite: 69].

---

## [cite_start]3. ARQUITECTURA DCBD MEDIANAMENTE ACOPLADA [cite: 70, 71]
* [cite_start]**Definición:** Ciertas tareas y algoritmos de descubrimiento forman parte del SGBD a través de procedimientos almacenados o funciones definidas por el usuario (FDUs)[cite: 72].
* [cite_start]**Integración por Procedimientos Almacenados:** Los algoritmos se encapsulan y corren en el mismo espacio de direccionamiento que el SGBD[cite: 86]. [cite_start]Al estar almacenados en la base de datos, un solo mensaje desencadena un conjunto de instrucciones SQL, lo que brinda un mejor desempeño[cite: 89, 90].
* [cite_start]**Integración por Funciones Definidas por el Usuario (FDUs):** Funciones creadas en lenguajes de propósito general cuyo ejecutable se almacena en el SGBD[cite: 91, 92]. [cite_start]Son más rápidas que los procedimientos almacenados, pero tienen la desventaja de un alto costo de desarrollo, ya que los algoritmos deben escribirse como FDUs[cite: 101, 102].
* [cite_start]**Ventajas y Retos:** Aprovecha la escalabilidad y administración del SGBD[cite: 74]. [cite_start]El reto es lograr un buen rendimiento, dado que el optimizador del SGBD no tiene estrategias para estas consultas específicas[cite: 75].

---

## [cite_start]4. ARQUITECTURA DCBD FUERTEMENTE ACOPLADA [cite: 108]
* [cite_start]**Definición:** Todas las tareas y algoritmos de descubrimiento se integran dentro del motor del SGBD como operaciones primitivas[cite: 109].
* [cite_start]**Ventajas y Limitaciones:** Resuelve los problemas de escalabilidad y rendimiento de las otras arquitecturas[cite: 111]. [cite_start]Sin embargo, la limitación es la reticencia de los desarrolladores a incluir algoritmos completos en las bases de datos debido a la competencia y las actualizaciones constantes de la investigación[cite: 112, 113].
* **Propuestas de Extensión del Lenguaje SQL:**
    * [cite_start]**DMQL:** Lenguaje con sintaxis similar a SQL que añade operadores para minería, pero fue implementado sobre DBMiner, el cual tiene arquitectura débilmente acoplada[cite: 125, 126, 133].
    * [cite_start]**M-SQL:** Extiende SQL con un operador unificado llamado MINE; se implementó en Data Mine, también bajo acoplamiento débil[cite: 128, 138].
    * [cite_start]**MINE RULE:** Operador para buscar reglas de asociación; su arquitectura divide el sistema en una interfaz de usuario, un *kernel* (con traductor, preprocesador, etc.) y el SGBD[cite: 143, 144, 152]. [cite_start]Aunque los autores la consideran fuertemente acoplada, la extracción se hace en un módulo independiente sobre el SGBD[cite: 176].
    * [cite_start]**NonStop SQL/MX:** Primitivas (como *transposition*, particionamiento y *sampling*) dentro de un motor SGBD paralelo[cite: 184, 185]. [cite_start]Es la única propuesta considerada verdaderamente acoplada de manera fuerte, aunque es específica para motores paralelos[cite: 194, 195].
    * [cite_start]**Microsoft OLE-DB para DM:** Proyecto para dotar de estándares de minería y lograr una futura integración fuerte con SQL Server[cite: 201, 202, 204].
    * [cite_start]**Proyecto de la Universidad del Valle:** Investigación para extender el motor relacional de POSTGRES con nuevas primitivas para soportar eficientemente el proceso, optimizando consultas mediante una álgebra relacional extendida[cite: 208, 209, 211].

---

## [cite_start]5. CONCLUSIONES [cite: 214]
* [cite_start]Es estrictamente necesario desarrollar métodos eficientes para extraer conocimiento debido al constante crecimiento de las bases de datos[cite: 215, 216].
* [cite_start]La mayoría de herramientas actuales funcionan con arquitecturas débilmente acopladas y son ineficientes computacionalmente[cite: 217].
* [cite_start]Actualmente no hay un SGBD relacional que logre hacer minería de forma completamente eficiente bajo una arquitectura fuertemente acoplada estándar[cite: 218].
* [cite_start]Aunque existen propuestas para añadir nuevos operadores SQL, se han implementado principalmente en sistemas débilmente acoplados[cite: 219].
* [cite_start]No hay un consenso general sobre las primitivas requeridas, por lo que la integración de sistemas DCBD y SGBD continuará siendo un área de investigación activa en el futuro[cite: 220, 222].