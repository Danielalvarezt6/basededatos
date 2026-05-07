"""
Genera las gráficas de comparación Extension PL/Python vs Weka
replicando la Figure 1 y Figure 2 del artículo:
  Vallejo-Cabrera et al., Rev. Fac. Ing., Vol. 34, No. 74 (2025)

Requiere haber ejecutado primero:
  1. benchmark_test1.py  → results/test1_extension_results.csv
  2. benchmark_weka.py   → results/test1_weka_results.csv
                           results/test2_weka_results.csv
  3. benchmark_test2.py  → results/test2_extension_results.csv
"""

import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})

os.makedirs("results/figures", exist_ok=True)

K_VALUES = list(range(2, 11))

# Colores fieles al artículo (azul = Extension, naranja = Weka)
COLOR_EXT  = "#1f77b4"   # azul
COLOR_WEKA = "#ff7f0e"   # naranja

MARKER_EXT  = "o"
MARKER_WEKA = "o"


# =====================================================
# FIGURE 1 — Test 1: variando registros
# =====================================================
def plot_figure1():
    csv_ext  = "results/test1_extension_results.csv"
    csv_weka = "results/test1_weka_results.csv"

    missing = [f for f in [csv_ext, csv_weka] if not os.path.exists(f)]
    if missing:
        print(f"[!] Archivos faltantes para Figure 1: {missing}")
        print("    Ejecuta primero benchmark_test1.py y benchmark_weka.py")
        return

    df_ext  = pd.read_csv(csv_ext)
    df_weka = pd.read_csv(csv_weka)

    sizes = sorted(df_ext["records"].unique())

    # Grid 3 × 3 (igual al artículo) — si hay menos de 9 tamaños, ajusta
    n_cols = 3
    n_rows = int(np.ceil(len(sizes) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten()

    for idx, size in enumerate(sizes):
        ax = axes[idx]

        ext_times  = []
        weka_times = []

        for k in K_VALUES:
            row_ext = df_ext[(df_ext["records"] == size) & (df_ext["k"] == k)]
            row_wk  = df_weka[(df_weka["records"] == size) & (df_weka["k"] == k)]

            ext_times.append(row_ext["kmeans_time_seconds"].values[0] if len(row_ext) else np.nan)
            weka_times.append(row_wk["total_time_seconds"].values[0]  if len(row_wk)  else np.nan)

        ax.plot(K_VALUES, ext_times,  color=COLOR_EXT,  marker=MARKER_EXT,
                linewidth=1.5, markersize=4, label="Extension")
        ax.plot(K_VALUES, weka_times, color=COLOR_WEKA, marker=MARKER_WEKA,
                linewidth=1.5, markersize=4, label="Weka")

        ax.set_title(f"number of records = {size:,}")
        ax.set_xlabel("number of clusters (k)")
        ax.set_ylabel("time in seconds")
        ax.set_xticks(K_VALUES)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left")

    # Ocultar celdas vacías
    for idx in range(len(sizes), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Figure 1. Performance comparison varying the number of records and the value of k",
        fontsize=10, y=1.01
    )
    plt.tight_layout()

    out_path = "results/figures/figure1_test1_records.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Figure 1 guardada en: {os.path.abspath(out_path)}")


# =====================================================
# FIGURE 2 — Test 2: variando atributos
# =====================================================
def plot_figure2():
    csv_ext  = "results/test2_extension_results.csv"
    csv_weka = "results/test2_weka_results.csv"

    missing = [f for f in [csv_ext, csv_weka] if not os.path.exists(f)]
    if missing:
        print(f"[!] Archivos faltantes para Figure 2: {missing}")
        print("    Ejecuta primero benchmark_test2.py y benchmark_weka.py")
        return

    df_ext  = pd.read_csv(csv_ext)
    df_weka = pd.read_csv(csv_weka)

    n_attrs_list = sorted(df_ext["attributes"].unique())

    n_cols = 3
    n_rows = int(np.ceil(len(n_attrs_list) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten()

    for idx, n_attrs in enumerate(n_attrs_list):
        ax = axes[idx]

        ext_times  = []
        weka_times = []

        for k in K_VALUES:
            row_ext = df_ext[(df_ext["attributes"] == n_attrs) & (df_ext["k"] == k)]
            row_wk  = df_weka[(df_weka["attributes"] == n_attrs) & (df_weka["k"] == k)]

            ext_times.append(row_ext["kmeans_time_seconds"].values[0] if len(row_ext) else np.nan)
            weka_times.append(row_wk["total_time_seconds"].values[0]  if len(row_wk)  else np.nan)

        ax.plot(K_VALUES, ext_times,  color=COLOR_EXT,  marker=MARKER_EXT,
                linewidth=1.5, markersize=4, label="Extension")
        ax.plot(K_VALUES, weka_times, color=COLOR_WEKA, marker=MARKER_WEKA,
                linewidth=1.5, markersize=4, label="Weka")

        ax.set_title(f"number of attributes = {n_attrs}")
        ax.set_xlabel("number of clusters (k)")
        ax.set_ylabel("time in seconds")
        ax.set_xticks(K_VALUES)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left")

    for idx in range(len(n_attrs_list), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Figure 2. Performance comparison varying the number of attributes and the value of k",
        fontsize=10, y=1.01
    )
    plt.tight_layout()

    out_path = "results/figures/figure2_test2_attributes.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Figure 2 guardada en: {os.path.abspath(out_path)}")


# =====================================================
# TABLA 6 — Resumen comparativo (replica Table 6 del artículo)
# =====================================================
def print_table6():
    print("\n" + "=" * 70)
    print("Table 6 — Comparative performance analysis: Extension vs Weka")
    print("=" * 70)

    for test, ext_csv, weka_csv, varying, fixed in [
        ("Test 1", "results/test1_extension_results.csv",
         "results/test1_weka_results.csv",
         "records", "attributes=11"),
        ("Test 2", "results/test2_extension_results.csv",
         "results/test2_weka_results.csv",
         "attributes", "records=21,000"),
    ]:
        if not os.path.exists(ext_csv) or not os.path.exists(weka_csv):
            print(f"[!] Faltan CSV para {test}")
            continue

        df_ext  = pd.read_csv(ext_csv)
        df_weka = pd.read_csv(weka_csv)

        time_col_ext  = "kmeans_time_seconds" if "kmeans_time_seconds" in df_ext.columns else "total_time_seconds"
        time_col_weka = "total_time_seconds"

        mean_ext  = df_ext[time_col_ext].mean()
        mean_weka = df_weka[time_col_weka].mean()
        speedup   = mean_weka / mean_ext if mean_ext > 0 else float("inf")

        print(f"\n  {test} (variando {varying}, {fixed})")
        print(f"    Media Extension : {mean_ext:.4f} s")
        print(f"    Media Weka      : {mean_weka:.4f} s")
        print(f"    Extension es {speedup:.1f}x más rápida que Weka (promedio)")


if __name__ == "__main__":
    print("=" * 70)
    print("  GENERANDO GRÁFICAS — Replica Figure 1 y Figure 2 del artículo")
    print("=" * 70)

    plot_figure1()
    plot_figure2()
    print_table6()

    print("\n[✓] Proceso completado. Revisa la carpeta results/figures/")
