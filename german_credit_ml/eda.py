# german_credit_ml/eda.py

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(input_path: Path, output_dir: Path):
    """
    Carga los datos limpios y genera visualizaciones EDA.
    """
    # Asegurarse de que el directorio de salida exista
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar datos
    print(f"Cargando datos limpios desde {input_path}...")
    df = pd.read_csv(input_path)
    
    # --- Generación de Gráficas ---
    
    print("Generando visualizaciones...")
    
    # 1. Histograma de variables numéricas
    numeric_cols = ['age', 'amount', 'duration']
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribución de {col.capitalize()}', fontsize=16)
        plt.xlabel(col.capitalize())
        plt.ylabel('Frecuencia')
        plt.tight_layout()
        
        # Guardar la gráfica
        output_file = output_dir / f"hist_{col}.png"
        plt.savefig(output_file)
        plt.close() # Cerrar la figura para liberar memoria
        print(f" -> Gráfica guardada en: {output_file}")

    # 2. Gráfica de barras de la variable objetivo
    plt.figure(figsize=(8, 6))
    sns.countplot(x='credit_risk', data=df)
    plt.title('Distribución del Riesgo Crediticio (Target)', fontsize=16)
    plt.xlabel('Riesgo Crediticio (0 = Malo, 1 = Bueno)')
    plt.ylabel('Conteo')
    plt.tight_layout()
    output_file = output_dir / "bar_credit_risk.png"
    plt.savefig(output_file)
    plt.close()
    print(f" -> Gráfica guardada en: {output_file}")
    
    # 3. Mapa de calor de correlaciones
    plt.figure(figsize=(16, 12))
    # Seleccionamos solo columnas numéricas para la correlación
    df_numeric = df.select_dtypes(include='number')
    corr = df_numeric.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=.5)
    plt.title('Mapa de Calor de Correlaciones', fontsize=18)
    plt.tight_layout()
    output_file = output_dir / "heatmap_correlation.png"
    plt.savefig(output_file)
    plt.close()
    print(f" -> Gráfica guardada en: {output_file}")

    print("\nAnálisis EDA completado.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ejecutar Análisis Exploratorio de Datos (EDA).")
    parser.add_argument("--input-data", type=str, required=True, help="Ruta al archivo CSV de datos limpios.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directorio donde se guardarán las gráficas.")
    
    args = parser.parse_args()
    
    run_eda(Path(args.input_data), Path(args.output_dir))

# --- Guardar resumen de datos para comparación ---
    
    # 1. Estadísticas descriptivas de variables numéricas
    desc_stats = df.describe().round(2)
    desc_file = Path("outputs_original") / "describe.csv"
    desc_file.parent.mkdir(parents=True, exist_ok=True)  # Crear carpeta si no existe
    desc_stats.to_csv(desc_file)
    print(f" -> Resumen estadístico guardado en: {desc_file}")

    # 2. Conteo de la variable objetivo
    target_counts = df['credit_risk'].value_counts().sort_index()
    target_file = Path("outputs_original") / "target_counts.csv"
    target_counts.to_csv(target_file, header=True)
    print(f" -> Conteo de la variable objetivo guardado en: {target_file}")
