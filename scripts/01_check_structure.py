#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verificador de estructura Cookiecutter con estados:
[PRESENTE], [FALTA-REQUERIDO], [FALTA-RECOMENDADO], [FALTA-OPCIONAL].

Ejecutar desde la raíz del repositorio.
"""

import os
import glob
from typing import List, Optional, Tuple

ROOT = os.path.abspath(os.curdir)
ROOT_NAME = os.path.basename(ROOT.rstrip(os.sep)) or "project-name"

# Niveles de importancia
REQUIRED = "requerido"
RECOMMENDED = "recomendado"
OPTIONAL = "opcional"

def exists(path: str) -> bool:
    return os.path.exists(os.path.join(ROOT, path))

def first_package_under_src() -> Optional[str]:
    candidates = [d for d in glob.glob("src/*") if os.path.isdir(d)]
    for d in sorted(candidates):
        if os.path.isfile(os.path.join(d, "__init__.py")):
            return os.path.basename(d)
    return None

def pyproject_or_requirements_setup() -> Tuple[bool, str]:
    """
    Regla compuesta (marcada como REQUERIDA):
    OK si existe pyproject.toml, o bien requirements.txt + setup.cfg
    """
    pyproject = exists("pyproject.toml")
    req_setup = exists("requirements.txt") and exists("setup.cfg")
    label = "pyproject.toml     (o requirements.txt + setup.cfg)"
    return (pyproject or req_setup, label)

class Node:
    def __init__(
        self,
        label: str,
        path: Optional[str] = None,
        children: Optional[List["Node"]] = None,
        importance: str = REQUIRED,            # requerido | recomendado | opcional
        optional_note: Optional[str] = None,   # texto corto al lado del label
        right_note: Optional[str] = None,      # nota a la derecha (columna fija)
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
            # Nodo visual (carpetas de nivel lógico, p.ej. project-name/), se muestra siempre
            return True
        return exists(self.path)

    def missing_tag(self) -> str:
        if self.path is None and self.exists_override is None:
            return ""  # nodos puramente visuales no se marcan
        if self.exists():
            return "[PRESENTE]"
        if self.importance == REQUIRED:
            return "[FALTA-REQUERIDO]"
        if self.importance == RECOMMENDED:
            return "[FALTA-RECOMENDADO]"
        return "[FALTA-OPCIONAL]"

def build_template_tree() -> Node:
    pkg = first_package_under_src() or "<package_name>"
    ok_req, label_req = pyproject_or_requirements_setup()

    return Node(f"{ROOT_NAME}/", path=None, importance=REQUIRED, children=[
        Node("README.md", "README.md", importance=REQUIRED),
        Node("LICENSE", "LICENSE", importance=RECOMMENDED, optional_note="(opcional pero recomendado)"),
        Node(".gitignore", ".gitignore", importance=REQUIRED),
        Node(".env.example", ".env.example", importance=RECOMMENDED, optional_note="(variables, sin secretos)"),

        # Dependencias: regla compuesta (requerido)
        Node(label_req, None, importance=REQUIRED, exists_override=ok_req),

        Node("Makefile", "Makefile", importance=RECOMMENDED, optional_note="(tareas comunes)"),

        Node("data/", "data", importance=REQUIRED, children=[
            Node("raw/", "data/raw", importance=REQUIRED, optional_note="(solo lectura)"),
            Node("interim/", "data/interim", importance=REQUIRED, optional_note="(steps intermedios)"),
            Node("processed/", "data/processed", importance=REQUIRED, optional_note="(dataset limpio para modeling)"),
        ]),

        Node("models/", "models", importance=RECOMMENDED, optional_note="(artefactos entrenados)"),
        Node("notebooks/", "notebooks", importance=RECOMMENDED, optional_note="(numerados: 0x-... con autor y propósito)"),

        Node("src/", "src", importance=REQUIRED, optional_note="(tu paquete de código)", children=[
            Node(f"{pkg}/", f"src/{pkg}", importance=REQUIRED, children=[
                Node("__init__.py", f"src/{pkg}/__init__.py", importance=REQUIRED),
                Node("data/", f"src/{pkg}/data", importance=RECOMMENDED, optional_note="(IO de datos)"),
                Node("features/", f"src/{pkg}/features", importance=RECOMMENDED, optional_note="(featurización)"),
                Node("models/", f"src/{pkg}/models", importance=RECOMMENDED, optional_note="(train, predict, evaluate)"),
                Node("utils/", f"src/{pkg}/utils", importance=RECOMMENDED, optional_note="(helpers)"),
            ])
        ]),

        Node("tests/", "tests", importance=RECOMMENDED, optional_note="(pytest)"),
        Node("reports/", "reports", importance=RECOMMENDED, optional_note="(figuras, tablas)"),

        Node("dvc.yaml", "dvc.yaml", importance=OPTIONAL, optional_note="(si usas DVC)"),
        Node("params.yaml", "params.yaml", importance=RECOMMENDED, optional_note="(hiperparámetros/config)"),
        Node("mlruns/", "mlruns", importance=OPTIONAL, optional_note="(si MLflow local)", right_note="← puede estar ignorado"),
        Node(".pre-commit-config.yaml", ".pre-commit-config.yaml", importance=RECOMMENDED, optional_note="(formato/linters)"),
    ])

def pad_to_column(text: str, col: int) -> str:
    if len(text) >= col:
        return text + " "
    return text + " " * (col - len(text))

def render_tree(node: Node, prefix: str = "", is_last: bool = True, right_note_col: int = 70):
    connector = "└─ " if is_last else "├─ "
    line_prefix = prefix + connector if prefix else ""
    base_label = node.label

    # Añadir notas cortas junto al label con una ligera alineación
    line = f"{line_prefix}{base_label}"
    if node.optional_note:
        line = pad_to_column(line, 40) + node.optional_note

    # Estado
    state = node.missing_tag()
    if state:
        line = pad_to_column(line, 65) + state

    # Nota a la derecha (columna fija)
    if node.right_note:
        line = pad_to_column(line, right_note_col) + node.right_note

    print(line)

    child_prefix = prefix + ("   " if is_last else "│  ")
    for i, child in enumerate(node.children):
        render_tree(child, prefix=child_prefix, is_last=(i == len(node.children) - 1), right_note_col=right_note_col)

def print_legend():
    print("\nLeyenda de estados:")
    print("  [PRESENTE]           Elemento encontrado en el repo")
    print("  [FALTA-REQUERIDO]    Elemento requerido ausente (debe corregirse)")
    print("  [FALTA-RECOMENDADO]  Elemento recomendado ausente (muy aconsejable)")
    print("  [FALTA-OPCIONAL]     Elemento opcional ausente (según necesidades)")

def main():
    tree = build_template_tree()
    render_tree(tree, prefix="", is_last=True)
    print_legend()

if __name__ == "__main__":
    main()
