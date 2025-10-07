# german_credit_ml/clean.py

# python -m german_credit_ml.clean --input data/raw/german_credit_modified.csv --output data/processed/german_credit_clean.csv


from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats.mstats import winsorize

LOGGER = logging.getLogger(__name__)

# --- Column Renaming and Type Definitions ---
COLUMN_MAPPING = {
    'laufkont': 'status', 'laufzeit': 'duration', 'moral': 'credit_history',
    'verw': 'purpose', 'hoehe': 'amount', 'sparkont': 'savings',
    'beszeit': 'employment_duration', 'rate': 'installment_rate',
    'famges': 'personal_status_sex', 'buerge': 'other_debtors',
    'wohnzeit': 'present_residence', 'verm': 'property', 'alter': 'age',
    'weitkred': 'other_installment_plans', 'wohn': 'housing',
    'bishkred': 'number_credits', 'beruf': 'job', 'pers': 'people_liable',
    'telef': 'telephone', 'gastarb': 'foreign_worker', 'kredit': 'credit_risk'
}

CATEGORIES_MAP = {
    "status": [1, 2, 3, 4], "credit_history": [0, 1, 2, 3, 4],
    "purpose": list(range(0, 11)), "savings": [1, 2, 3, 4, 5],
    "employment_duration": [1, 2, 3, 4, 5], "installment_rate": [1, 2, 3, 4],
    "personal_status_sex": [1, 2, 3, 4], "other_debtors": [1, 2, 3],
    "present_residence": [1, 2, 3, 4], "property": [1, 2, 3, 4],
    "other_installment_plans": [1, 2, 3], "housing": [1, 2, 3],
    "job": [1, 2, 3, 4], "telephone": [1, 2], "foreign_worker": [1, 2],
}

NUMERIC_COLS = ["duration", "amount", "age"]
TARGET_COL = "credit_risk"

# ----------------------------
# Helper Functions
# ----------------------------

def load_yaml(path: Path | str) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ----------------------------
# Core Cleaning Logic
# ----------------------------

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Applies the full cleaning pipeline to a single dataframe."""
    
    df = df.rename(columns=COLUMN_MAPPING)

    if 'mixed_type_col' in df.columns:
        df = df.drop(columns=['mixed_type_col'])

    LOGGER.info("Step 1: Forcing all columns to numeric type...")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    LOGGER.info("Step 2: Normalizing and filtering target variable 'credit_risk'...")
    df[TARGET_COL] = df[TARGET_COL].replace({1.0: 1, 2.0: 0}).astype(float)
    df = df[df[TARGET_COL].isin([0.0, 1.0])].copy()

    LOGGER.info("Step 3: Imputing missing values using the data's own median/mode...")
    for col in df.columns:
        if df[col].isnull().any():
            if col in NUMERIC_COLS:
                impute_value = df[col].median()
                df[col] = df[col].fillna(impute_value)
            elif col in CATEGORIES_MAP:
                impute_value = df[col].mode()[0]
                df[col] = df[col].fillna(impute_value)
    
    df = df.fillna(0)

    LOGGER.info("Step 4: Applying Winsorization to numeric columns...")
    for col in NUMERIC_COLS:
        df[col] = winsorize(df[col], limits=[0.01, 0.01])
        
    LOGGER.info("Step 5: Enforcing final data types and categories...")
    for col in NUMERIC_COLS:
        df[col] = df[col].astype("int64")
        
    for col, valid_cats in CATEGORIES_MAP.items():
        df[col] = df[col].clip(lower=min(valid_cats), upper=max(valid_cats))
        df[col] = df[col].astype("category")

    df[TARGET_COL] = df[TARGET_COL].astype("int64")
    other_int_cols = ['number_credits', 'people_liable']
    for col in other_int_cols:
         if col in df.columns:
              df[col] = df[col].astype('int64')

    return df

def final_validation(df: pd.DataFrame):
    """Performs final checks on the cleaned dataframe."""
    LOGGER.info("Performing final validation...")
    assert df.isnull().sum().sum() == 0, "Validation failed: Null values still exist."
    uniq_targets = set(df[TARGET_COL].unique())
    assert uniq_targets.issubset({0, 1}), f"Validation failed: Target column contains non-binary values: {uniq_targets}"
    LOGGER.info("Validation successful: No nulls and target is binary.")

# ----------------------------
# Main Execution Block
# ----------------------------
def run_clean(input_path: Path, output_path: Path):
    LOGGER.info(f"Reading raw data from: {input_path}")
    df = pd.read_csv(input_path)
    
    df_cleaned = clean_dataframe(df)
    
    final_validation(df_cleaned)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_csv(output_path, index=False)
    LOGGER.info(f"Cleaned data saved to: {output_path} (Shape: {df_cleaned.shape})")

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cleaning script for the German Credit dataset.")
    p.add_argument("--config", type=str, help="Path to the YAML configuration file (e.g., params.yaml).")
    p.add_argument("--input", type=str, help="Path to the raw input CSV file (overrides config).")
    p.add_argument("--output", type=str, help="Path to save the cleaned output CSV file (overrides config).")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
    return p

def main():
    args = build_argparser().parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    if args.config:
        cfg = load_yaml(args.config)
        input_path = Path(args.input or cfg["data_load"]["raw"])
        output_path = Path(args.output or cfg["data_load"]["processed"])
    elif args.input and args.output:
        input_path = Path(args.input)
        output_path = Path(args.output)
    else:
        raise SystemExit("Error: You must provide either --config or both --input and --output.")

    run_clean(input_path, output_path)

if __name__ == "__main__":
    main()