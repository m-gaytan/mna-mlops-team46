#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verificador de estructura Cookiecutter Data Science v2 con estados:
[PRESENTE], [FALTA-REQUERIDO], [FALTA-RECOMENDADO], [FALTA-OPCIONAL].

Diferencia clave v2:
- Ya no se usa una carpeta intermedia src/.
- El paquete principal del proyecto vive directamente en la raíz con el nombre
  {{ cookiecutter.module_name }} (en tu caso, german_credit_ml).
  Dentro de ese paquete van submódulos como data/, features/, models/, utils/, etc.

Uso:
  python 01_check_structure.py \
    --project-root "C:\\dev\\mna-mlops-team46" \
    --module-name german_credit_ml

También puedes usar variables de entorno:
  PROJECT_ROOT, MODULE_NAME
"""

import os
import glob
import argparse
from typing import List, Optional, Tuple
import sys
import io

# Forzar stdout/stderr a UTF-8 en Windows para caracteres como ├─, └─, etc.
if sys.stdout.encoding is None or "cp125" in sys.stdout.encoding.lower():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding is None or "cp125" in sys.stderr.encoding.lower():
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
# -------------------- CLI / ENV --------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Validador de estructura Cookiecutter Data Science v2")
    parser.add_argument(
        "--project-root",
        default=os.getenv("PROJECT_ROOT", r"C:\dev\mna-mlops-team46"),
        help="Ruta absoluta del proyecto (raíz del repo)",
    )
    parser.add_argument(
        "--module-name",
        default=os.getenv("MODULE_NAME", "german_credit_ml"),
        help="Nombre del paquete Python al nivel raíz (Cookiecutter v2)",
    )
    return parser.parse_args()

ARGS = parse_args()

# Normaliza y cambia a la raíz del proyecto
ROOT = os.path.abspath(os.path.expanduser(ARGS.project_root))
if not os.path.isdir(ROOT):
    raise SystemExit(f"[ERROR] Project root no existe: {ROOT}")
os.chdir(ROOT)

ROOT_NAME = os.path.basename(ROOT.rstrip(os.sep)) or "project-name"
MODULE_NAME = ARGS.module_name.strip() or "<package_name>"

# -------------------- Constantes --------------------
REQUIRED = "requerido"
RECOMMENDED = "recomendado"
OPTIONAL = "opcional"

# -------------------- Utilidades --------------------
def exists(path: str) -> bool:
    return os.path.exists(os.path.join(ROOT, path))

def find_top_level_module() -> Optional[str]:
    """
    Cookiecutter Data Science v2:
    ya no hay src/, así que buscamos un paquete directamente en la raíz
    que tenga __init__.py.
    Si el usuario nos da --module-name, lo validamos primero.
    """
    candidate = os.path.join(MODULE_NAME, "__init__.py")
    if os.path.isfile(candidate):
        return MODULE_NAME

    # fallback: intentar detectar automáticamente el primero que parezca paquete
    candidates = [d for d in glob.glob("*") if os.path.isdir(d)]
    for d in sorted(candidates):
        if os.path.isfile(os.path.join(d, "__init__.py")):
            return os.path.basename(d)
    return None

def get_package_name() -> str:
    """
    Devuelve el nombre real del paquete raíz para inspección.
    """
    auto = find_top_level_module()
    if auto:
        return auto
    return MODULE_NAME or "<package_name>"

def pyproject_or_requirements_setup() -> Tuple[bool, str]:
    """
    Regla compuesta (REQUIRED):
      OK si existe pyproject.toml, o bien requirements.txt + setup.cfg
    """
    pyproject = exists("pyproject.toml")
    req_setup = exists("requirements.txt") and exists("setup.cfg")
    label = "pyproject.toml     (o requirements.txt + setup.cfg)"
    return (pyproject or req_setup, label)

# -------------------- Modelo de árbol --------------------
class Node:
    def __init__(
        self,
        label: str,
        path: Optional[str] = None,
        children: Optional[List["Node"]] = None,
        importance: str = REQUIRED,            # requerido | recomendado | opcional
        optional_note: Optional[str] = None,   # texto corto junto al label
        right_note: Optional[str] = None,      # nota alineada a la derecha
        exists_override: Optional[bool] = None # para reglas compuestas
    ):
        self.label = label
        self.path = path
        self.children = children or []
        self.importance = importance
        self.optional_note = optional_note
        self.right_note = right_note
        self.exists_override = exists_override

    def exists(self) -> bool:
        if self.exists_override is not None:
            return self.exists_override
        if self.path is None:
            # Nodo visual puro (e.g. raíz lógica): siempre lo mostramos como contexto
            return True
        return exists(self.path)

    def state_tag(self) -> str:
        # carpeta raíz lógica no tiene estado
        if self.path is None and self.exists_override is None:
            return ""
        if self.exists():
            return "[PRESENTE]"
        if self.importance == REQUIRED:
            return "[FALTA-REQUERIDO]"
        if self.importance == RECOMMENDED:
            return "[FALTA-RECOMENDADO]"
        return "[FALTA-OPCIONAL]"

def build_template_tree() -> Node:
    pkg = get_package_name()
    ok_req, label_req = pyproject_or_requirements_setup()

    # En Cookiecutter DS v2 el paquete principal está al nivel raíz:
    # <repo>/
    #   README.md
    #   data/
    #   notebooks/
    #   reports/
    #   models/
    #   <package_name>/        <-- en tu caso german_credit_ml
    #       __init__.py
    #       data/
    #       features/
    #       models/
    #       utils/
    #
    # Nota: ya NO aparece src/

    return Node(f"{ROOT_NAME}/", path=None, importance=REQUIRED, children=[
        Node("README.md", "README.md", importance=REQUIRED),
        Node("LICENSE", "LICENSE", importance=RECOMMENDED, optional_note="(opcional pero recomendado)"),
        Node(".gitignore", ".gitignore", importance=REQUIRED),
        Node(".env.example", ".env.example", importance=RECOMMENDED, optional_note="(variables, sin secretos)"),

        # Dependencias / instalación
        Node(label_req, None, importance=REQUIRED, exists_override=ok_req),

        Node("Makefile", "Makefile", importance=RECOMMENDED, optional_note="(tareas comunes)"),

        Node("data/", "data", importance=REQUIRED, children=[
            Node("raw/", "data/raw", importance=REQUIRED, optional_note="(solo lectura)"),
            Node("interim/", "data/interim", importance=REQUIRED, optional_note="(steps intermedios)"),
            Node("processed/", "data/processed", importance=REQUIRED, optional_note="(dataset limpio para modeling)"),
        ]),

        Node("models/", "models", importance=RECOMMENDED, optional_note="(artefactos entrenados)"),
        Node("notebooks/", "notebooks", importance=RECOMMENDED, optional_note="(numerados: 0x-... con autor y propósito)"),
        Node("reports/", "reports", importance=RECOMMENDED, optional_note="(figuras, tablas)"),

        # Paquete principal del proyecto (ya no bajo src/)
        Node(f"{pkg}/", f"{pkg}", importance=REQUIRED, optional_note="(paquete Python principal del proyecto)", children=[
            Node("__init__.py", f"{pkg}/__init__.py", importance=REQUIRED),
            Node("data/", f"{pkg}/data", importance=RECOMMENDED, optional_note="(IO de datos)"),
            Node("features/", f"{pkg}/features", importance=RECOMMENDED, optional_note="(featurización)"),
            Node("models/", f"{pkg}/models", importance=RECOMMENDED, optional_note="(train, predict, evaluate)"),
            Node("utils/", f"{pkg}/utils", importance=RECOMMENDED, optional_note="(helpers/utilidades compartidas)"),
        ]),

        Node("tests/", "tests", importance=RECOMMENDED, optional_note="(pytest)"),

        # Artefactos de MLOps
        Node("dvc.yaml", "dvc.yaml", importance=OPTIONAL, optional_note="(si usas DVC)"),
        Node("params.yaml", "params.yaml", importance=RECOMMENDED, optional_note="(hiperparámetros/config)"),
        Node("mlruns/", "mlruns", importance=OPTIONAL, optional_note="(si MLflow local)", right_note="← puede estar ignorado"),
        Node(".pre-commit-config.yaml", ".pre-commit-config.yaml", importance=RECOMMENDED, optional_note="(formato/linters)"),
    ])

def pad_to_column(text: str, col: int) -> str:
    return text if len(text) >= col else text + " " * (col - len(text))

def render_tree(node: Node, prefix: str = "", is_last: bool = True, right_note_col: int = 70):
    connector = "└─ " if is_last else "├─ "
    line_prefix = prefix + connector if prefix else ""
    line = f"{line_prefix}{node.label}"

    if node.optional_note:
        line = pad_to_column(line, 40) + node.optional_note

    state = node.state_tag()
    if state:
        line = pad_to_column(line, 70) + state

    if node.right_note:
        line = pad_to_column(line, right_note_col) + node.right_note

    print(line)

    child_prefix = prefix + ("   " if is_last else "│  ")
    for i, child in enumerate(node.children):
        render_tree(
            child,
            prefix=child_prefix,
            is_last=(i == len(node.children) - 1),
            right_note_col=right_note_col,
        )

def print_legend():
    print("\nLeyenda de estados:")
    print("  [PRESENTE]           Elemento encontrado en el repo")
    print("  [FALTA-REQUERIDO]    Elemento requerido ausente (debe corregirse)")
    print("  [FALTA-RECOMENDADO]  Elemento recomendado ausente (muy aconsejable)")
    print("  [FALTA-OPCIONAL]     Elemento opcional ausente (según necesidades)")

def print_header(title: str):
    print("\n" + title)
    print("-" * len(title))

def main():
    print_header("1) Estructuración de Proyectos con Cookiecutter Data Science v2")
    tree = build_template_tree()
    render_tree(tree, prefix="", is_last=True)
    print_legend()

if __name__ == "__main__":
    main()
