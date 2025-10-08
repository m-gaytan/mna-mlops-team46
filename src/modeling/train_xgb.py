# src/modeling/train_xgb.py
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, confusion_matrix, ConfusionMatrixDisplay,
                             roc_auc_score)

from xgboost import XGBClassifier

import mlflow
from mlflow.models import infer_signature
import getpass, socket, subprocess, os
from datetime import datetime


def main(args):
    # Usa tracking local si no quieres mezclarte con el servidor de otro
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"file:{os.getcwd()}/mlruns"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "fase1_modelado_equipo46"))

    run_name = f"train_xgb_{getpass.getuser()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        
        # --- antes o después de fit() ---
        mlflow.set_tag("author", getpass.getuser())
        mlflow.set_tag("host", socket.gethostname())
        mlflow.set_tag("git_branch", subprocess.check_output(["git","rev-parse","--abbrev-ref","HEAD"]).decode().strip())
        mlflow.set_tag("git_commit", subprocess.check_output(["git","rev-parse","--short","HEAD"]).decode().strip())
        mlflow.set_tag("dvc_stage", "train")
        # si quieres, etiqueta el dataset
        mlflow.set_tag("dataset", "data/processed/german_credit_clean.csv")
        # ... entrena, calcula métricas, loggea modelo/figuras ...


    print("==== PASO 1: Carga y Preparación de Datos ====")
    data_path = Path(args.input_data).resolve()
    print(f"\nCargando: {data_path}")
    df = pd.read_csv(data_path)
    print(f"\nDatos cargados con {df.shape[0]} filas y {df.shape[1]} columnas.")

    TARGET = "credit_risk"  # según tu dataset limpio
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print("Datos divididos en conjuntos de entrenamiento y prueba.")

    print("\n==== PASO 2: Entrenamiento del Modelo ====")
    params = {
        "n_estimators": 150,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": 42,
        "tree_method": "hist",         # rápido local
        "use_label_encoder": False,    # evita warning viejo
        "n_jobs": -1,
    }
    print("\nEntrenando un modelo XGBoost con los siguientes parámetros:")
    print(params)

    clf = XGBClassifier(**params)
    clf.fit(X_train, y_train)

    print("\n==== PASO 3: Evaluación, Guardado y Registro ====")
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"\nF1-Score en el conjunto de prueba: {f1:.4f}")
    print(f"ROC-AUC en el conjunto de prueba: {auc:.4f}")

    # Matriz de confusión → figura
    fig_out = Path(args.fig_output).resolve()
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Matriz de confusión (XGB)")
    plt.savefig(fig_out, bbox_inches="tight")
    plt.close(fig)
    print(f"→ Matriz de confusión guardada en: {fig_out}")

    # Guardar modelo (pickle)
    import joblib
    model_out = Path(args.model_output).resolve()
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_out)
    print(f"→ Modelo guardado en formato .pkl: {model_out}")

    # MLflow local (sin cuelgues)
    mlruns_dir = Path.cwd() / "mlruns"
    mlflow.set_tracking_uri(f"file:{mlruns_dir}")
    mlflow.set_experiment(args.mlflow_experiment)

    # Firma e input_example
    X_example = X_test.iloc[:5].copy()
    signature = infer_signature(X_example, clf.predict_proba(X_example)[:, 1])

    # Requisitos fijos (evita pip freeze largo)
    import sklearn, cloudpickle
    pip_reqs = [
        f"scikit-learn=={sklearn.__version__}",
        f"cloudpickle=={cloudpickle.__version__}",
        "xgboost",
        "numpy","pandas","matplotlib","mlflow","joblib"
    ]

    # Run
    if mlflow.active_run(): mlflow.end_run()
    with mlflow.start_run(run_name="xgboost_baseline"):
        mlflow.log_metric("f1", float(f1))
        mlflow.log_metric("roc_auc", float(auc))
        mlflow.log_params(params)

        # artefactos
        mlflow.log_artifact(str(fig_out), artifact_path="figures")

        # modelo
        try:
            mlflow.sklearn.log_model(
                sk_model=clf, name="model",
                input_example=X_example, signature=signature,
                pip_requirements=pip_reqs
            )
        except TypeError:
            mlflow.sklearn.log_model(
                sk_model=clf, artifact_path="model",
                input_example=X_example, signature=signature,
                pip_requirements=pip_reqs
            )

    print("\nActualizando lock de DVC si este script se ejecuta como stage...")
    print("Hecho.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-data", required=True)
    p.add_argument("--model-output", default="models/xgboost_model.pkl")
    p.add_argument("--fig-output",   default="reports/figures/training/confusion_matrix.png")
    p.add_argument("--mlflow-experiment", default="fase1_modelado_equipo46")
    args = p.parse_args()
    main(args)
