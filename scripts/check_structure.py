#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verificador de estructura Cookiecutter con salida tipo árbol.

Imprime la plantilla esperada y marca con [FALTA] lo que no exista
en el proyecto actual. Ejecutar desde la raíz del repositorio.

Salida ejemplo:

project-name/
├─ README.md
├─ LICENSE            (opcional pero recomendado)
...
"""

import os
import glob
from typing import List, Optional, Tuple

ROOT = os.path.abspath(os.curdir)
ROOT_NAME = os.path.basename(ROOT.rstrip(os.sep)) or "project-name"


def exists(path: str) -> bool:
    """Verifica existencia relativa a la raíz del repo."""
    return os.path.exists(os.path.join(ROOT, path))


def first_package_under_src() -> Optional[str]:
    """Detecta el primer paquete válido dentro de src/ (directorio con __init__.py)"""
    candidates = [d for d in glob.glob("src/*") if os.path.isdir(d)]
    for d in sorted(candidates):
        if os.path.isfile(os.path.join(d, "__init__.py")):
            return os.path.basename(d)
    return None


def pyproject_or_requirements_setup() -> Tuple[bool, str]:
    """
    Regla de equivalencia:
      - OK si existe pyproject.toml
      - o si existen requirements.txt + setup.cfg
    Devuelve (ok, etiqueta_para_mostrar)
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
        optional_note: Optional[str] = None,
        note: Optional[str] = None,
        exists_override: Optional[bool] = None,
    ):
        """
        label: texto mostrado (no necesariamente el path real)
        path: ruta relativa para verificar existencia (None => solo visual)
        children: nodos hijos
        optional_note: texto opcional mostrado después del label (p.ej. '(opcional...)')
        note: texto adicional alineado a la derecha (p.ej. '← puede estar ignorado')
        exists_override: fuerza el estado de existencia (para reglas compuestas)
        """
        self.label = label
        self.path = path
        self.children = children or []
        self.optional_note = optional_note
        self.note = note
        self.exists_override = exists_override

    def exists(self) -> bool:
        if self.exists_override is not None:
            return self.exists_override
        if self.path is None:
            # Nodo puramente visual, considerar existente
            return True
        return exists(self.path)


def build_template_tree() -> Node:
    # Detectar nombre de paquete bajo src o placeholder
    pkg = first_package_under_src() or "<package_name>"

    # Regla especial pyproject/requirements+setup
    ok_req, label_req = pyproject_or_requirements_setup()

    tree = Node(f"{ROOT_NAME}/", path=None, children=[
        Node("README.md", "README.md"),
        Node("LICENSE", "LICENSE", optional_note="(opcional pero recomendado)"),
        Node(".gitignore", ".gitignore"),
        Node(".env.example", ".env.example", optional_note="(variables, sin secretos)"),
        Node(label_req, None, exists_override=ok_req),  # regla compuesta
        Node("Makefile", "Makefile", optional_note="(tareas comunes)"),

        Node("data/", "data", children=[
            Node("raw/", "data/raw", optional_note="(solo lectura)"),
            Node("interim/", "data/interim", optional_note="(steps intermedios)"),
            Node("processed/", "data/processed", optional_note="(dataset limpio para modeling)"),
        ]),

        Node("models/", "models", optional_note="(artefactos entrenados)"),
        Node("notebooks/", "notebooks", optional_note="(numerados: 0x-... con autor y propósito)"),

        Node("src/", "src", optional_note="(tu paquete de código)", children=[
            Node(f"{pkg}/", f"src/{pkg}", children=[
                Node("__init__.py", f"src/{pkg}/__init__.py"),
                Node("data/", f"src/{pkg}/data", optional_note="(IO de datos)"),
                Node("features/", f"src/{pkg}/features", optional_note="(featurización)"),
                Node("models/", f"src/{pkg}/models", optional_note="(train, predict, evaluate)"),
                Node("utils/", f"src/{pkg}/utils", optional_note="(helpers)"),
            ])
        ]),

        Node("tests/", "tests", optional_note="(pytest)"),
        Node("reports/", "reports", optional_note="(figuras, tablas)"),

        Node("dvc.yaml", "dvc.yaml", optional_note="(si usas DVC)"),
        Node("params.yaml", "params.yaml", optional_note="(hiperparámetros/config)"),
        Node("mlruns/", "mlruns", optional_note="(si MLflow local)", note="← puede estar ignorado"),
        Node(".pre-commit-config.yaml", ".pre-commit-config.yaml", optional_note="(formato/linters)"),
    ])
    return tree


def pad_to_column(text: str, col: int) -> str:
    """Espacia hasta la columna 'col' (1-indexed)."""
    if len(text) >= col:
        return text + " "
    return text + " " * (col - len(text))


def render_tree(node: Node, prefix: str = "", is_last: bool = True, right_note_col: int = 60):
    """
    Imprime el árbol con conectores, marcando [FALTA] si un nodo requerido no existe.
    Las notas opcionales se muestran al lado del nombre.
    Las 'right notes' (node.note) se alinean a la columna indicada.
    """
    connector = "└─ " if is_last else "├─ "
    line_prefix = prefix + connector if prefix else ""
    label = node.label

    # Construir etiqueta con opcionales
    if node.optional_note:
        # Alinear simple entre label y nota opcional
        label_shown = f"{label}"
        # El missing se marca después
    else:
        label_shown = label

    # Estado de existencia
    exists_flag = node.exists()

    # Marcar faltantes (no marcamos los puramente visuales: sin path y sin override)
    missing_tag = ""
    if node.path is not None or node.exists_override is not None:
        if not exists_flag:
            missing_tag = " [FALTA]"

    # Ensamblar línea base
    base = f"{line_prefix}{label_shown}"
    if node.optional_note:
        base = pad_to_column(base, 40) + node.optional_note  # ligera alineación visual

    base += missing_tag

    # Nota a la derecha (ej. '← puede estar ignorado')
    if node.note:
        base = pad_to_column(base, right_note_col) + node.note

    print(base)

    # Hijos
    child_prefix = prefix + ("   " if is_last else "│  ")
    for i, child in enumerate(node.children):
        last = (i == len(node.children) - 1)
        render_tree(child, prefix=child_prefix, is_last=last, right_note_col=right_note_col)


def main():
    tree = build_template_tree()
    # Encabezado
    render_tree(tree, prefix="", is_last=True)


if __name__ == "__main__":
    main()
