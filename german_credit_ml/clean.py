# german_credit_ml/clean.py
from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)


# ----------------------------
# Utilidades de configuración
# ----------------------------
def load_yaml(path: Path | str) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró config: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ----------------------------
# Lógica de limpieza
# ----------------------------
CAT_COLS: List[str] = [
    "status", "credit_history", "purpose", "savings",
    "employment_duration", "installment_rate", "personal_status_sex",
    "other_debtors", "present_residence", "property",
    "other_installment_plans", "housing", "number_credits",
    "job", "people_liable", "telephone", "foreign_worker",
]

INT_COLS: List[str] = ["amount", "age", "duration", "number_credits", "people_liable", "credit_risk"]

# Si ya conoces los rangos de categorías (P2), puedes usarlos aquí.
CATEGORIES_MAP: Dict[str, List[int]] = {
    "status": [1, 2, 3, 4],
    "credit_history": [0, 1, 2, 3, 4],
    "purpose": list(range(0, 10)),           # 0..9
    "savings": [1, 2, 3, 4, 5],
    "employment_duration": [0, 1, 2, 3, 4],
    "installment_rate": [1, 2, 3, 4],
    "personal_status_sex": [1, 2, 3, 4],
    "other_debtors": [1, 2, 3],
    "present_residence": [1, 2, 3, 4],
    "property": [1, 2, 3, 4],
    "other_installment_plans": [1, 2, 3],
    "housing": [1, 2, 3],
    "number_credits": [1, 2, 3, 4],
    "job": [1, 2, 3, 4],
    "people_liable": [1, 2],
    "telephone": [1, 2],
    "foreign_worker": [1, 2],
}

def normalize_credit_risk(df: pd.DataFrame) -> pd.DataFrame:
    if "credit_risk" in df.columns:
        df["credit_risk"] = pd.to_numeric(df["credit_risk"], errors="coerce").replace({2.0: 1.0, 3.0: 0.0})
    return df

def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    # numéricas a int (coerce primero)
    for col in INT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # categóricas a numérico primero (porque vienen codificadas)
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def impute_and_enforce(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Imputación para INT_COLS (mediana redondeada) y casteo final a int64
    for col in [c for c in INT_COLS if c in df.columns]:
        med = df[col].median(skipna=True)
        med_int = int(round(med)) if pd.notna(med) else 0
        df[col] = df[col].fillna(med_int).astype("int64")

    # 2) Imputación para categóricas con moda y restricción a categorías válidas
    for col, cats in CATEGORIES_MAP.items():
        if col in df.columns:
            series = df[col]
            if series.notna().any():
                mode_val = int(series.mode(dropna=True).iloc[0])
            else:
                mode_val = cats[0]
            series = series.fillna(mode_val)
            # clamp al rango válido
            series = series.clip(lower=min(cats), upper=max(cats)).astype("int64")
            # guardar como category con las categorías definidas
            df[col] = pd.Categorical(series, categories=cats, ordered=False)

    return df

def basic_validation(df: pd.DataFrame):
    # Chequeos simples
    assert "credit_risk" in df.columns, "Falta columna target 'credit_risk'."
    assert set(df["credit_risk"].unique()).issubset({0, 1}), "credit_risk debe ser 0/1."
    # ejemplo: no nulos
    if df.isnull().sum().sum() > 0:
        nulos = df.isnull().sum().sort_values(ascending=False).head(10)
        raise ValueError(f"Aún hay nulos tras la limpieza:\n{nulos}")

# ----------------------------
# Pipeline
# ----------------------------
def run_clean(input_path: Path, output_path: Path):
    LOGGER.info(f"Leyendo datos de: {input_path}")
    df = pd.read_csv(input_path)

    LOGGER.info("Normalizando target credit_risk…")
    df = normalize_credit_risk(df)

    LOGGER.info("Casteando tipos base…")
    df = cast_types(df)

    LOGGER.info("Imputando y fijando dominios de categorías…")
    df = impute_and_enforce(df)

    LOGGER.info("Validando…")
    basic_validation(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    LOGGER.info(f"Guardado en: {output_path}  (filas={len(df)})")


# ----------------------------
# CLI
# ----------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Limpieza del German Credit")
    p.add_argument("--config", type=str, default=None, help="Ruta a config YAML (opcional)")
    p.add_argument("--input", type=str, default=None, help="CSV de entrada (si no usas YAML)")
    p.add_argument("--output", type=str, default=None, help="CSV de salida (si no usas YAML)")
    p.add_argument("--log-level", type=str, default="INFO", help="Nivel de logging")
    return p

def main():
    args = build_argparser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.config:
        cfg = load_yaml(args.config)
        input_path = Path(cfg["paths"]["raw"])
        output_path = Path(cfg["paths"]["processed"])
    else:
        if not args.input or not args.output:
            raise SystemExit("Debes pasar --config o bien --input y --output.")
        input_path = Path(args.input)
        output_path = Path(args.output)

    run_clean(input_path, output_path)

if __name__ == "__main__":
    main()
