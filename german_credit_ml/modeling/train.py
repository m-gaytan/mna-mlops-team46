# german_credit_ml/modeling/train.py

import argparse
import json
from pathlib import Path
import warnings
import pickle

import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# (El resto de las importaciones se mantienen igual)
# from scipy.stats import randint, uniform

warnings.filterwarnings("ignore", category=FutureWarning)

def train_model(input_data: Path, model_output: Path, metrics_output: Path, plots_output: Path, params: dict):
    """
    Entrena un modelo XGBoost y registra el experimento con un nombre específico.
    """
    # --- AÑADE ESTA LÍNEA PARA NOMBRAR TU EXPERIMENTO ---
    mlflow.set_experiment("German Credit XGBoost")

    with mlflow.start_run():
        
        print("\n" + "="*50)
        print(" PASO 1: Carga y Preparación de Datos ".center(50, "="))
        print("="*50)
        
        df = pd.read_csv(input_data)
        # ... (el resto del código de carga y preparación es igual)
        X = df.drop(columns='credit_risk')
        y = df['credit_risk']
        X = pd.get_dummies(X, drop_first=True)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=params.get('test_size', 0.2),
            random_state=params.get('random_state', 42),
            stratify=y
        )
        
        print("\n" + "="*50)
        print(" PASO 2: Entrenamiento del Modelo ".center(50, "="))
        print("="*50)

        fixed_params = {
            'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.1,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'use_label_encoder': False, 'eval_metric': 'logloss',
            'random_state': params.get('random_state', 42)
        }
        
        print("Entrenando un modelo XGBoost con parámetros fijos...")
        model = xgb.XGBClassifier(**fixed_params)
        model.fit(X_train, y_train)
        
        print("\n" + "="*50)
        print(" PASO 3: Evaluación, Guardado y Registro ".center(50, "="))
        print("="*50)
        
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        f1_score = report['1']['f1-score']
        
        print(f"F1-Score en el conjunto de prueba: {f1_score:.4f}")
        
        # Generar y guardar la matriz de confusión
        plots_output.mkdir(parents=True, exist_ok=True)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Malo (0)', 'Bueno (1)'], yticklabels=['Malo (0)', 'Bueno (1)'])
        plt.title('Matriz de Confusión', fontsize=16)
        plt.ylabel('Etiqueta Verdadera')
        plt.xlabel('Etiqueta Predicha')
        confusion_matrix_path = plots_output / "confusion_matrix.png"
        plt.savefig(confusion_matrix_path)
        plt.close()

        # Registro en MLflow
        print("\nRegistrando experimento en MLflow...")
        mlflow.log_params(fixed_params)
        mlflow.log_metric("f1_score_test", f1_score)
        mlflow.xgboost.log_model(model, "xgboost-model")
        mlflow.log_artifact(confusion_matrix_path, "plots")
        
        # Guardado de archivos para DVC
        with open(model_output, 'wb') as f:
            pickle.dump(model, f)
        
        metrics = {'f1_score_test': f1_score, 'params': fixed_params}
        with open(metrics_output, 'w') as f:
            json.dump(metrics, f, indent=4, default=str)

# (El bloque if __name__ == '__main__': se mantiene igual)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Entrenar un modelo XGBoost rápido.")
    # ... (argumentos)
    parser.add_argument("--input-data", type=str, required=True)
    parser.add_argument("--model-output", type=str, required=True)
    parser.add_argument("--metrics-output", type=str, required=True)
    parser.add_argument("--plots-output", type=str, required=True)
    
    args = parser.parse_args()
    
    train_params = {'test_size': 0.2, 'random_state': 42}
    
    train_model(
        input_data=Path(args.input_data),
        model_output=Path(args.model_output),
        metrics_output=Path(args.metrics_output),
        plots_output=Path(args.plots_output),
        params=train_params
    )