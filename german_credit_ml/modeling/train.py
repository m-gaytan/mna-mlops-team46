# german_credit_ml/modeling/train.py

import argparse
import json
from pathlib import Path
import warnings
import pickle
import datetime

import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)

def train_model(input_data: Path, model_output: Path, metrics_output: Path, plots_output: Path, params: dict):
    mlflow.set_experiment("German Credit XGBoost")
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"run_{now}"

    with mlflow.start_run(run_name=run_name):
        # ... (PASO 1 y 2: Carga y Entrenamiento - sin cambios) ...
        # ...
        
        print("\n" + "="*50)
        print(" PASO 3: Evaluación y Registro ".center(50, "="))
        print("="*50)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # --- Calcular Métricas ---
        report = classification_report(y_test, y_pred, output_dict=True)
        f1 = report['1']['f1-score']
        accuracy = report['accuracy']
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # NUEVO: Calcular la métrica personalizada "Bad Rate"
        # Es el porcentaje de predicciones que son igual a 0 (Malo)
        bad_rate = np.mean(y_pred == 0)

        print(f"F1-Score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"Bad Rate: {bad_rate:.4f}") # NUEVO

        # ... (Generación de Gráficas de Evaluación - sin cambios) ...
        # ...
        
        # --- Registro en MLflow ---
        print("\nRegistrando experimento en MLflow...")
        mlflow.log_params(model.get_params())
        
        mlflow.log_metric("f1_score_test", f1)
        mlflow.log_metric("accuracy_test", accuracy)
        mlflow.log_metric("auc_test", auc)
        mlflow.log_metric("bad_rate_test", bad_rate) # NUEVO
        
        # (El resto del registro de artefactos en MLflow se mantiene igual)
        # ...
        
        # --- Guardado de archivos para DVC ---
        with open(model_output, 'wb') as f:
            pickle.dump(model, f)
        
        # NUEVO: Añadir la métrica personalizada al archivo JSON
        metrics = {
            'f1_score_test': f1,
            'accuracy_test': accuracy,
            'auc_test': auc,
            'bad_rate_test': bad_rate,
            'params': model.get_params()
        }
        with open(metrics_output, 'w') as f:
            json.dump(metrics, f, indent=4, default=str)

# (El bloque if __name__ == '__main__': se mantiene igual)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Entrenar un modelo XGBoost rápido.")
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