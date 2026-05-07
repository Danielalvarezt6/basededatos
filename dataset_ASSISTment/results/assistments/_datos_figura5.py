import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
FIGURAS_DIR = BASE_DIR / 'figuras'
FIGURAS_DIR.mkdir(exist_ok=True)

# 1. Cargar los datos desde los archivos CSV
df_ext = pd.read_csv(BASE_DIR / 'prueba1_extension.csv')
df_weka = pd.read_csv(BASE_DIR / 'prueba1_weka.csv')
df_sk = pd.read_csv(BASE_DIR / 'prueba1_sklearn.csv')

# 2. Agrupar por 'registros' y calcular el promedio de los tiempos
ext_mean = pd.DataFrame(df_ext.groupby('registros')[['tiempo_carga_s', 'tiempo_kmeans_s']].mean()).reset_index()
weka_mean = pd.DataFrame(df_weka.groupby('registros')[['tiempo_carga_s', 'tiempo_kmeans_s']].mean()).reset_index()
sk_mean = pd.DataFrame(df_sk.groupby('registros')[['tiempo_carga_s', 'tiempo_kmeans_s']].mean()).reset_index()

# 3. Calcular el máximo global para usar la misma escala en los tres subplots
y_max_global = max(
    (ext_mean['tiempo_carga_s'] + ext_mean['tiempo_kmeans_s']).max(),
    (weka_mean['tiempo_carga_s'] + weka_mean['tiempo_kmeans_s']).max(),
    (sk_mean['tiempo_carga_s'] + sk_mean['tiempo_kmeans_s']).max(),
)

datasets = [
    (ext_mean,  'Extensión PL/Python', ('steelblue',    'navy')),
    (weka_mean, 'Weka',                ('lightcoral',   'crimson')),
    (sk_mean,   'Python sklearn',      ('lightgreen',   'forestgreen')),
]

etiquetas_x = [f'{int(r/1000)}K' if r < 1_000_000 else '1M'
               for r in ext_mean['registros']]
x = np.arange(len(etiquetas_x))
ancho = 0.55

# 4. Crear figura con 3 subplots compartiendo el eje Y
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

fig.suptitle(
    'Figura 5 — Tiempo de carga de datos vs. Tiempo de ejecución K-Means\n'
    'Comparativa por herramienta (k = 5 grupos, Dataset ASSISTments)',
    fontsize=11, fontweight='bold'
)

for ax, (df, titulo, (color_carga, color_exec)) in zip(axes, datasets):
    barras_carga = ax.bar(x, df['tiempo_carga_s'],
                          width=ancho, label='Carga de datos',
                          color=color_carga, alpha=0.85)
    ax.bar(x, df['tiempo_kmeans_s'],
           width=ancho, bottom=df['tiempo_carga_s'],
           label='Ejecución K-Means', color=color_exec, alpha=0.85)

    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel('Número de registros', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas_x)
    ax.set_ylim(0, y_max_global * 1.08)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].set_ylabel('Tiempo (s)', fontsize=9)

plt.tight_layout()

# 5. Guardar y mostrar
salida = FIGURAS_DIR / 'figura5_carga_vs_ejecucion.png'
plt.savefig(salida, dpi=150, bbox_inches='tight')
print(f'Figura guardada en: {salida}')
plt.show()
