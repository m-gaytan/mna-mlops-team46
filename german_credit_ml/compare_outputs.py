import os
import filecmp

# Carpetas de outputs
folder1 = "outputs_original"
folder2 = "outputs_v2"

# Lista los archivos en folder1
files1 = os.listdir(folder1)

# Compara cada archivo con su par en folder2
for f in files1:
    f1 = os.path.join(folder1, f)
    f2 = os.path.join(folder2, f)
    
    if os.path.isfile(f1) and os.path.isfile(f2):
        if filecmp.cmp(f1, f2, shallow=False):
            print(f"{f}: IGUALES ✅")
        else:
            print(f"{f}: DIFERENTES ❌")
    else:
        print(f"{f}: NO EXISTE EN UNA DE LAS CARPETAS ⚠️")
