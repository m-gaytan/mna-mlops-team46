# german_credit_ml/modeling/train.py

import argparse
import json
from pathlib import Path
import warnings
import pickle

# Importaciones de ML y visualización
import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import randint, uniform

# Ignorar advertencias futuras para una salida más limpia
warnings.filterwarnings("ignore", category=FutureWarning)

def train_model(input_data: Path, model_output: Path, metrics_output: Path, plots_output: Path, params: dict):
    """
    Entrena un modelo XGBoost, registra el experimento con MLflow, y genera
    una matriz de confusión y un archivo .pkl del modelo.
    """
    # Inicia un "Run" de MLflow para registrar todo en un solo experimento
    with mlflow.start_run():
        
        print("\n" + "="*50)
        print(" PASO 1: Carga y Preparación de Datos ".center(50, "="))
        print("="*50)
        
        df = pd.read_csv(input_data)
        print(f"Datos cargados con {df.shape[0]} filas y {df.shape[1]} columnas.")
        
        # Separar features (X) y target (y)
        X = df.drop(columns='credit_risk')
        y = df['credit_risk']
        
        # Convertir variables categóricas a dummies (One-Hot Encoding)
        X = pd.get_dummies(X, drop_first=True)
        
        # Dividir datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=params.get('test_size', 0.2),
            random_state=params.get('random_state', 42),
            stratify=y  # Mantener proporción de clases en la división
        )
        print("Datos divididos en conjuntos de entrenamiento y prueba.")
        
        print("\n" + "="*50)
        print(" PASO 2: Búsqueda de Hiperparámetros ".center(50, "="))
        print("="*50)

        # Definir el clasificador y el espacio de búsqueda de parámetros
        clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=params.get('random_state', 42))
        
        param_dist = {
            'n_estimators': randint(50, 500),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4)
        }
        
        # Configurar la búsqueda aleatoria con validación cruzada
        # El parámetro 'verbose=2' mostrará el progreso en la consola.
        random_search = RandomizedSearchCV(
            clf, param_distributions=param_dist,
            n_iter=params.get('n_iter_search', 20),
            scoring='f1',
            cv=5,
            random_state=params.get('random_state', 42),
            n_jobs=-1, # Usar todos los procesadores disponibles
            verbose=2
        )
        
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        
        print("\n" + "="*50)
        print(" PASO 3: Evaluación, Guardado y Registro ".center(50, "="))
        print("="*50)
        
        print(f"Mejores hiperparámetros encontrados: {random_search.best_params_}")
        
        # Evaluar el mejor modelo en el conjunto de prueba
        y_pred = best_model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        f1_score = report['1']['f1-score'] # F1-score para la clase positiva (1 = 'Bueno')
        
        print(f"F1-Score en el conjunto de prueba: {f1_score:.4f}")
        
        # Generar y guardar la matriz de confusión
        plots_output.mkdir(parents=True, exist_ok=True)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Malo (0)', 'Bueno (1)'], yticklabels=['Malo (0)', 'Bueno (1)'])
        plt.title('Matriz de Confusión - Mejor Modelo', fontsize=16)
        plt.ylabel('Etiqueta Verdadera')
        plt.xlabel('Etiqueta Predicha')
        confusion_matrix_path = plots_output / "confusion_matrix.png"
        plt.savefig(confusion_matrix_path)
        plt.close()
        print(f" -> Matriz de confusión guardada en: {confusion_matrix_path}")

        # --- Registro de todo en MLflow ---
        print("\nRegistrando experimento en MLflow...")
        mlflow.log_params(random_search.best_params_)
        mlflow.log_metric("f1_score_test", f1_score)
        mlflow.log_metric("accuracy_test", report['accuracy'])
        mlflow.xgboost.log_model(best_model, "xgboost-model")
        mlflow.log_artifact(confusion_matrix_path, "plots")
        
        # --- Guardado de archivos para DVC ---
        # Guardar el modelo como .pkl
        model_output.parent.mkdir(parents=True, exist_ok=True)
        with open(model_output, 'wb') as f:
            pickle.dump(best_model, f)
        print(f" -> Modelo guardado en formato .pkl en: {model_output}")

        # Guardar las métricas como .json
        metrics = {'f1_score_test': f1_score, 'best_params': random_search.best_params_}
        with open(metrics_output, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f" -> Métricas guardadas en: {metrics_output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Entrenar modelo XGBoost con MLflow y DVC.")
    parser.add_argument("--input-data", type=str, required=True, help="Ruta al CSV de datos limpios.")
    parser.add_argument("--model-output", type=str, required=True, help="Ruta para guardar el modelo .pkl.")
    parser.add_argument("--metrics-output", type=str, required=True, help="Ruta para guardar las métricas en JSON.")
    parser.add_argument("--plots-output", type=str, required=True, help="Directorio para guardar las gráficas de evaluación.")
    
    args = parser.parse_args()
    
    # DVC se encargará de leer y pasar los parámetros desde params.yaml
    print("Cargando parámetros desde params.yaml...")
    # (Este bloque es solo para ilustración, DVC maneja los parámetros en el pipeline real)
    train_params = {
        'test_size': 0.2,
        'random_state': 42,
        'n_iter_search': 25
    }
    
    train_model(
        input_data=Path(args.input_data),
        model_output=Path(args.model_output),
        metrics_output=Path(args.metrics_output),
        plots_output=Path(args.plots_output),
        params=train_params
    )