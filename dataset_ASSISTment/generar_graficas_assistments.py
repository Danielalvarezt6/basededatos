"""
Genera gráficas y tabla comparativa en ESPAÑOL para el cartel académico.
Dataset: ASSISTments — Perfiles de comportamiento de estudiantes

Compara 3 herramientas:
  1. Extensión PL/Python (PostgreSQL)
  2. Weka (SimpleKMeans)
  3. Python scikit-learn

Figuras generadas:
  - Figura 1: Prueba 1 — tiempo vs. número de registros (panel por herramienta)
  - Figura 2: Prueba 2 — tiempo vs. número de atributos (panel por herramienta)
  - Figura 3: Comparativa directa de las 3 herramientas (k fijo)

Tablas exportadas en CSV con columnas en español.

Salidas en: results/assistments/figuras/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

DIRECTORIO_RESULTADOS = "results/assistments"
DIRECTORIO_FIGURAS    = "results/assistments/figuras"

# CSV de entrada
CSV_P1_EXT    = os.path.join(DIRECTORIO_RESULTADOS, "prueba1_extension.csv")
CSV_P1_WEKA   = os.path.join(DIRECTORIO_RESULTADOS, "prueba1_weka.csv")
CSV_P1_SKLEARN = os.path.join(DIRECTORIO_RESULTADOS, "prueba1_sklearn.csv")

CSV_P2_EXT    = os.path.join(DIRECTORIO_RESULTADOS, "prueba2_extension.csv")
CSV_P2_WEKA   = os.path.join(DIRECTORIO_RESULTADOS, "prueba2_weka.csv")
CSV_P2_SKLEARN = os.path.join(DIRECTORIO_RESULTADOS, "prueba2_sklearn.csv")

# Colores por herramienta
COLOR_EXT    = "#1f77b4"   # azul
COLOR_WEKA   = "#d62728"   # rojo
COLOR_SKLEARN = "#2ca02c"  # verde

# Configuración de herramientas
HERRAMIENTAS = [
    {
        "nombre":    "Extensión PL/Python\n(PostgreSQL)",
        "csv_p1":    CSV_P1_EXT,
        "csv_p2":    CSV_P2_EXT,
        "col_tiempo": "tiempo_kmeans_s",
        "color":     COLOR_EXT,
        "marcador":  "o",
        "etiqueta":  "Extensión PL/Python",
    },
    {
        "nombre":    "Weka\n(SimpleKMeans)",
        "csv_p1":    CSV_P1_WEKA,
        "csv_p2":    CSV_P2_WEKA,
        "col_tiempo": "tiempo_total_s",
        "color":     COLOR_WEKA,
        "marcador":  "s",
        "etiqueta":  "Weka",
    },
    {
        "nombre":    "Python\nscikit-learn",
        "csv_p1":    CSV_P1_SKLEARN,
        "csv_p2":    CSV_P2_SKLEARN,
        "col_tiempo": "tiempo_kmeans_s",
        "color":     COLOR_SKLEARN,
        "marcador":  "^",
        "etiqueta":  "Python sklearn",
    },
]

MARCADORES_K = ["o", "s", "^", "D", "v", "P", "*", "X", "h"]
K_VALORES    = list(range(2, 11))


# =====================================================
# UTILIDADES
# =====================================================
def cargar_csv(ruta: str) -> pd.DataFrame | None:
    if not os.path.exists(ruta):
        return None
    return pd.read_csv(ruta)


def etiqueta_registros(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)


# =====================================================
# FIGURA 1 — Cuadrícula estilo paper: un panel por volumen de registros
#             X = k (grupos), líneas = herramientas   (réplica Figura 1 del artículo)
# =====================================================
def figura1_prueba_registros():
    """
    Replica el estilo de la Figura 1 del artículo:
    cuadrícula 3×N donde cada panel es un volumen de registros,
    el eje X es k y cada línea es una herramienta.
    """
    # Recopilar tamaños disponibles en todos los CSV
    tamanos = set()
    for herr in HERRAMIENTAS:
        df = cargar_csv(herr["csv_p1"])
        if df is not None:
            tamanos.update(df["registros"].unique().tolist())
    tamanos = sorted(tamanos)

    if not tamanos:
        print("[!] Sin datos para Figura 1.")
        return

    n_cols = 3
    n_rows = (len(tamanos) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 4.5 * n_rows),
                             squeeze=False)
    fig.suptitle(
        "Figura 1 — Tiempo de respuesta total variando el número de registros y el valor de k\n"
        "Dataset ASSISTments — Extensión PL/Python · Weka · Python sklearn",
        fontsize=13, fontweight="bold",
    )

    for idx, tamano in enumerate(tamanos):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        for herr in HERRAMIENTAS:
            df = cargar_csv(herr["csv_p1"])
            if df is None:
                continue
            df_s  = df[df["registros"] == tamano].sort_values("num_grupos")
            # Usar tiempo de respuesta total (carga + K-Means) para comparación justa
            col_t = "tiempo_respuesta_s" if "tiempo_respuesta_s" in df_s.columns else \
                    ("tiempo_kmeans_s" if "tiempo_kmeans_s" in df_s.columns else "tiempo_total_s")
            if df_s.empty:
                continue
            ax.plot(df_s["num_grupos"], df_s[col_t],
                    marker=herr["marcador"], color=herr["color"],
                    linewidth=1.8, label=herr["etiqueta"])

        ax.set_title(f"número de registros = {etiqueta_registros(tamano)}", fontsize=10)
        ax.set_xlabel("número de grupos (k)")
        ax.set_ylabel("tiempo de respuesta (s)")
        ax.set_xticks(K_VALORES)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    # Ocultar paneles vacíos
    for idx in range(len(tamanos), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    ruta = os.path.join(DIRECTORIO_FIGURAS, "figura1_prueba_registros.png")
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    print(f"[✓] Figura 1 guardada: {ruta}")
    plt.close()


# =====================================================
# FIGURA 2 — Cuadrícula estilo paper: un panel por número de atributos
#             X = k (grupos), líneas = herramientas   (réplica Figura 2 del artículo)
# =====================================================
def figura2_prueba_atributos():
    """
    Mismo estilo que Figura 1 pero para la prueba de atributos:
    un panel por subconjunto de atributos, eje X = k, líneas = herramientas.
    """
    n_attrs_vals = set()
    for herr in HERRAMIENTAS:
        df = cargar_csv(herr["csv_p2"])
        if df is not None:
            n_attrs_vals.update(df["num_atributos"].unique().tolist())
    n_attrs_vals = sorted(n_attrs_vals)

    if not n_attrs_vals:
        print("[!] Sin datos para Figura 2.")
        return

    n_cols = min(3, len(n_attrs_vals))
    n_rows = (len(n_attrs_vals) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 4.5 * n_rows),
                             squeeze=False)
    fig.suptitle(
        "Figura 2 — Tiempo de respuesta total variando el número de atributos y el valor de k\n"
        "Dataset ASSISTments — Extensión PL/Python · Weka · Python sklearn",
        fontsize=13, fontweight="bold",
    )

    for idx, n_attrs in enumerate(n_attrs_vals):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        for herr in HERRAMIENTAS:
            df = cargar_csv(herr["csv_p2"])
            if df is None:
                continue
            df_a  = df[df["num_atributos"] == n_attrs].sort_values("num_grupos")
            col_t = "tiempo_respuesta_s" if "tiempo_respuesta_s" in df_a.columns else \
                    ("tiempo_kmeans_s" if "tiempo_kmeans_s" in df_a.columns else "tiempo_total_s")
            if df_a.empty:
                continue
            ax.plot(df_a["num_grupos"], df_a[col_t],
                    marker=herr["marcador"], color=herr["color"],
                    linewidth=1.8, label=herr["etiqueta"])

        ax.set_title(f"número de atributos = {n_attrs}", fontsize=10)
        ax.set_xlabel("número de grupos (k)")
        ax.set_ylabel("tiempo de respuesta (s)")
        ax.set_xticks(K_VALORES)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    for idx in range(len(n_attrs_vals), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    ruta = os.path.join(DIRECTORIO_FIGURAS, "figura2_prueba_atributos.png")
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    print(f"[✓] Figura 2 guardada: {ruta}")
    plt.close()


# =====================================================
# FIGURA 3 — Tiempo de RESPUESTA: cuadrícula por registros
#             Igual que Fig 1 pero usando tiempo_respuesta_s
# =====================================================
def figura3_tiempo_respuesta_registros():
    tamanos = set()
    for herr in HERRAMIENTAS:
        df = cargar_csv(herr["csv_p1"])
        if df is not None and "tiempo_respuesta_s" in df.columns:
            tamanos.update(df["registros"].unique().tolist())
    tamanos = sorted(tamanos)

    if not tamanos:
        print("[!] Sin columna 'tiempo_respuesta_s'. Vuelve a ejecutar los benchmarks.")
        return

    n_cols = 3
    n_rows = (len(tamanos) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 4.5 * n_rows),
                             squeeze=False)
    fig.suptitle(
        "Figura 3 — Tiempo de RESPUESTA total (carga + K-Means) variando registros y k\n"
        "Dataset ASSISTments — Extensión PL/Python · Weka · Python sklearn",
        fontsize=13, fontweight="bold",
    )

    for idx, tamano in enumerate(tamanos):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        for herr in HERRAMIENTAS:
            df = cargar_csv(herr["csv_p1"])
            if df is None or "tiempo_respuesta_s" not in df.columns:
                continue
            df_s = df[df["registros"] == tamano].sort_values("num_grupos")
            if df_s.empty:
                continue
            ax.plot(df_s["num_grupos"], df_s["tiempo_respuesta_s"],
                    marker=herr["marcador"], color=herr["color"],
                    linewidth=1.8, label=herr["etiqueta"])

        ax.set_title(f"número de registros = {etiqueta_registros(tamano)}", fontsize=10)
        ax.set_xlabel("número de grupos (k)")
        ax.set_ylabel("tiempo de respuesta (s)")
        ax.set_xticks(K_VALORES)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    for idx in range(len(tamanos), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    ruta = os.path.join(DIRECTORIO_FIGURAS, "figura3_tiempo_respuesta_registros.png")
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    print(f"[✓] Figura 3 guardada: {ruta}")
    plt.close()


# =====================================================
# FIGURA 4 — Tiempo de RESPUESTA: cuadrícula por atributos
# =====================================================
def figura4_tiempo_respuesta_atributos():
    n_attrs_vals = set()
    for herr in HERRAMIENTAS:
        df = cargar_csv(herr["csv_p2"])
        if df is not None and "tiempo_respuesta_s" in df.columns:
            n_attrs_vals.update(df["num_atributos"].unique().tolist())
    n_attrs_vals = sorted(n_attrs_vals)

    if not n_attrs_vals:
        print("[!] Sin columna 'tiempo_respuesta_s' en Prueba 2.")
        return

    n_cols = min(3, len(n_attrs_vals))
    n_rows = (len(n_attrs_vals) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 4.5 * n_rows),
                             squeeze=False)
    fig.suptitle(
        "Figura 4 — Tiempo de RESPUESTA total (carga + K-Means) variando atributos y k\n"
        "Dataset ASSISTments — Extensión PL/Python · Weka · Python sklearn",
        fontsize=13, fontweight="bold",
    )

    for idx, n_attrs in enumerate(n_attrs_vals):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        for herr in HERRAMIENTAS:
            df = cargar_csv(herr["csv_p2"])
            if df is None or "tiempo_respuesta_s" not in df.columns:
                continue
            df_a = df[df["num_atributos"] == n_attrs].sort_values("num_grupos")
            if df_a.empty:
                continue
            ax.plot(df_a["num_grupos"], df_a["tiempo_respuesta_s"],
                    marker=herr["marcador"], color=herr["color"],
                    linewidth=1.8, label=herr["etiqueta"])

        ax.set_title(f"número de atributos = {n_attrs}", fontsize=10)
        ax.set_xlabel("número de grupos (k)")
        ax.set_ylabel("tiempo de respuesta (s)")
        ax.set_xticks(K_VALORES)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    for idx in range(len(n_attrs_vals), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    ruta = os.path.join(DIRECTORIO_FIGURAS, "figura4_tiempo_respuesta_atributos.png")
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    print(f"[✓] Figura 4 guardada: {ruta}")
    plt.close()


# =====================================================
# TABLA RESUMEN (exportada en CSV y mostrada en consola)
# =====================================================
def tabla_resumen():
    print("\n" + "=" * 85)
    print("TABLA RESUMEN — Comparativa de rendimiento (segundos)")
    print("Dataset: ASSISTments | Extensión PL/Python · Weka · Python sklearn")
    print("=" * 85)

    # P8: forzar tiempo_respuesta_s — único tiempo verdaderamente equivalente entre las 3 herramientas
    COL_COMPARACION = "tiempo_respuesta_s"

    # ── Tabla 1: Prueba 1, k=5 ──
    csvs_p1 = [(h["etiqueta"], cargar_csv(h["csv_p1"])) for h in HERRAMIENTAS]
    tamanos = sorted({s for _, df in csvs_p1 if df is not None for s in df["registros"].unique()})

    if tamanos:
        print(f"\n--- Prueba 1: {COL_COMPARACION} por número de registros (k = 5 grupos) ---")
        filas = []
        for s in tamanos:
            fila = {"Registros": etiqueta_registros(s)}
            for nombre, df in csvs_p1:
                if df is None:
                    fila[nombre + " (s)"] = "—"
                    continue
                col_usar = COL_COMPARACION if COL_COMPARACION in df.columns else "tiempo_total_s"
                sub = df[(df["registros"] == s) & (df["num_grupos"] == 5)]
                fila[nombre + " (s)"] = round(sub[col_usar].mean(), 4) if len(sub) else "—"
            filas.append(fila)
        tbl1 = pd.DataFrame(filas).set_index("Registros")
        print(tbl1.to_string())
        tbl1.to_csv(os.path.join(DIRECTORIO_RESULTADOS, "tabla_resumen_prueba1.csv"),
                    encoding="utf-8")

    # ── Tabla 2: Prueba 2, k=5 ──
    csvs_p2 = [(h["etiqueta"], cargar_csv(h["csv_p2"])) for h in HERRAMIENTAS]
    n_attrs = sorted({a for _, df in csvs_p2 if df is not None for a in df["num_atributos"].unique()})

    if n_attrs:
        print(f"\n--- Prueba 2: {COL_COMPARACION} por número de atributos (k = 5 grupos) ---")
        filas2 = []
        for a in n_attrs:
            fila = {"Atributos": a}
            for nombre, df in csvs_p2:
                if df is None:
                    fila[nombre + " (s)"] = "—"
                    continue
                col_usar = COL_COMPARACION if COL_COMPARACION in df.columns else "tiempo_total_s"
                sub = df[(df["num_atributos"] == a) & (df["num_grupos"] == 5)]
                fila[nombre + " (s)"] = round(sub[col_usar].mean(), 4) if len(sub) else "—"
            filas2.append(fila)
        tbl2 = pd.DataFrame(filas2).set_index("Atributos")
        print(tbl2.to_string())
        tbl2.to_csv(os.path.join(DIRECTORIO_RESULTADOS, "tabla_resumen_prueba2.csv"),
                    encoding="utf-8")

    print("\n[✓] Tablas exportadas en:", os.path.abspath(DIRECTORIO_RESULTADOS))


# =====================================================
# FIGURA 5 — Tiempo de carga vs tiempo de ejecución K-Means
# =====================================================
def figura5_carga_vs_ejecucion():
    """
    Barras apiladas: para cada herramienta muestra cuánto es carga de datos
    y cuánto es ejecución pura del K-Means (k=5, registros seleccionados).
    Demuestra la ventaja del procesamiento en-base de datos (sin transferencia).
    """
    registros_mostrar = [10_000, 100_000, 500_000, 1_000_000]
    k_fijo = 5

    datos = {}
    for herr in HERRAMIENTAS:
        df = cargar_csv(herr["csv_p1"])
        if df is None or "tiempo_carga_s" not in df.columns:
            continue
        df_k = df[df["num_grupos"] == k_fijo]
        cargas, ejecuciones, etiquetas = [], [], []
        for r in registros_mostrar:
            sub = df_k[df_k["registros"] == r]
            if len(sub):
                cargas.append(sub["tiempo_carga_s"].mean())
                ejecuciones.append(sub["tiempo_kmeans_s"].mean())
                etiquetas.append(etiqueta_registros(r))
        if cargas:
            datos[herr["etiqueta"]] = {
                "color": herr["color"],
                "cargas": cargas,
                "ejecuciones": ejecuciones,
                "etiquetas": etiquetas,
            }

    if not datos:
        print("[!] Sin columna 'tiempo_carga_s'. Vuelve a ejecutar los benchmarks.")
        return

    n_herr = len(datos)
    fig, axes = plt.subplots(1, n_herr, figsize=(6 * n_herr, 6), sharey=False)
    if n_herr == 1:
        axes = [axes]

    fig.suptitle(
        "Figura 5 — Tiempo de carga de datos vs. Tiempo de ejecución K-Means\n"
        f"Comparativa por herramienta (k = {k_fijo} grupos, Dataset ASSISTments)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    for ax, (nombre, d) in zip(axes, datos.items()):
        x = np.arange(len(d["etiquetas"]))
        ax.bar(x, d["cargas"],     label="Carga de datos",     color=d["color"], alpha=0.5)
        ax.bar(x, d["ejecuciones"], label="Ejecución K-Means", color=d["color"],
               alpha=0.9, bottom=d["cargas"])
        ax.set_xticks(x)
        ax.set_xticklabels(d["etiquetas"])
        ax.set_xlabel("Número de registros")
        ax.set_ylabel("Tiempo (s)")
        ax.set_title(nombre)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    ruta = os.path.join(DIRECTORIO_FIGURAS, "figura5_carga_vs_ejecucion.png")
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    print(f"[✓] Figura 5 guardada: {ruta}")
    plt.close()


# =====================================================
# FIGURA 6 — Tiempo de respuesta total (carga + ejecución)
# =====================================================
def figura6_tiempo_respuesta():
    """
    Compara el tiempo de RESPUESTA TOTAL (carga + K-Means) de las 3 herramientas.
    Es el tiempo real que experimenta el usuario desde que solicita hasta que recibe resultado.
    """
    k_fijo = 5
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Figura 6 — Tiempo de respuesta total al usuario (carga de datos + K-Means)\n"
        "Comparativa: Extensión PL/Python · Weka · Python sklearn  (k = 5 grupos)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # Panel izquierdo: vs registros
    ax1 = axes[0]
    tamanos = set()
    for herr in HERRAMIENTAS:
        df = cargar_csv(herr["csv_p1"])
        if df is not None and "tiempo_respuesta_s" in df.columns:
            tamanos.update(df["registros"].unique().tolist())
    tamanos = sorted(tamanos)

    for herr in HERRAMIENTAS:
        df = cargar_csv(herr["csv_p1"])
        if df is None or "tiempo_respuesta_s" not in df.columns:
            continue
        df_k = df[df["num_grupos"] == k_fijo].sort_values("registros")
        vals = [df_k[df_k["registros"] == s]["tiempo_respuesta_s"].mean()
                if s in df_k["registros"].values else np.nan for s in tamanos]
        ax1.plot(range(len(tamanos)), vals, marker=herr["marcador"],
                 color=herr["color"], linewidth=2, label=herr["etiqueta"])

    ax1.set_xticks(range(len(tamanos)))
    ax1.set_xticklabels([etiqueta_registros(s) for s in tamanos], rotation=35, ha="right")
    ax1.set_xlabel("Número de registros")
    ax1.set_ylabel("Tiempo de respuesta (s)")
    ax1.set_title("Prueba 1 — Variando registros")
    ax1.set_yscale("log")
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Panel derecho: vs atributos
    ax2 = axes[1]
    n_attrs_vals = set()
    for herr in HERRAMIENTAS:
        df = cargar_csv(herr["csv_p2"])
        if df is not None and "tiempo_respuesta_s" in df.columns:
            n_attrs_vals.update(df["num_atributos"].unique().tolist())
    n_attrs_vals = sorted(n_attrs_vals)

    for herr in HERRAMIENTAS:
        df = cargar_csv(herr["csv_p2"])
        if df is None or "tiempo_respuesta_s" not in df.columns:
            continue
        df_k = df[df["num_grupos"] == k_fijo].sort_values("num_atributos")
        xs = [a for a in n_attrs_vals if a in df_k["num_atributos"].values]
        ys = [df_k[df_k["num_atributos"] == a]["tiempo_respuesta_s"].mean() for a in xs]
        ax2.plot(xs, ys, marker=herr["marcador"], color=herr["color"],
                 linewidth=2, label=herr["etiqueta"])

    ax2.set_xlabel("Número de atributos")
    ax2.set_ylabel("Tiempo de respuesta (s)")
    ax2.set_title("Prueba 2 — Variando atributos")
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(2))

    plt.tight_layout()
    ruta = os.path.join(DIRECTORIO_FIGURAS, "figura6_tiempo_respuesta.png")
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    print(f"[✓] Figura 6 guardada: {ruta}")
    plt.close()


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    os.makedirs(DIRECTORIO_FIGURAS, exist_ok=True)
    print("=" * 70)
    print("  GENERANDO GRÁFICAS Y TABLA — Dataset ASSISTments (3 herramientas)")
    print("=" * 70)

    figura1_prueba_registros()          # estilo paper: cuadrícula por registros, k en eje X
    figura2_prueba_atributos()          # estilo paper: cuadrícula por atributos, k en eje X
    figura3_tiempo_respuesta_registros() # mismo estilo pero con tiempo de respuesta total
    figura4_tiempo_respuesta_atributos() # mismo estilo pero con tiempo de respuesta total
    figura5_carga_vs_ejecucion()        # barras: carga vs ejecución por herramienta
    figura6_tiempo_respuesta()          # resumen en 2 paneles
    tabla_resumen()

    print("\n[✓] Listo. Revisa:", os.path.abspath(DIRECTORIO_FIGURAS))
