import os, sys, glob, textwrap
ROOT = os.path.abspath(os.curdir)
REQUIRED_DIRS = ["src","nocleantebooks","models","reports","tests","data/raw","data/interim","data/processed"]
REQUIRED_FILES = ["README.md",".gitignore"]
RECOMMENDED = ["pyproject.toml","requirements.txt","Makefile","params.yaml",".pre-commit-config.yaml"]
def ex(p): return os.path.exists(os.path.join(ROOT,p))
missing = [p for p in REQUIRED_FILES + REQUIRED_DIRS if not ex(p)]
reco    = [p for p in RECOMMENDED if not ex(p)]
pkg_dirs = [d for d in glob.glob("src/*") if os.path.isdir(d) and os.path.isfile(os.path.join(d,"__init__.py"))]
print("\n=== CHECKLIST COOKIECUTTER ===")
print("OK críticos completos." if not missing else "Faltantes críticos:"); [print(" -",p) for p in missing]
print("\nOK recomendados presentes." if not reco else "Recomendados ausentes:"); [print(" -",p) for p in reco]
print("\nPaquetes en src/:", [os.path.basename(d) for d in pkg_dirs] or "Ninguno")
print(textwrap.dedent("""
Sugerencias:
 - Numerar notebooks (01-, 02-, 03-) con autor/propósito.
 - pyproject.toml o requirements{,-dev}.txt para dependencias.
 - Makefile con tareas: data, train, evaluate, lint, test.
 - DVC para versionar datos y remoto S3; params.yaml para rutas/semillas/hparams.
"""))