#!/usr/bin/env python
"""
Arma model/ddpm-cifar10-32/ (listo para usar y para subir a GitHub) a partir de los archivos del modelo
que usted ya descomprimió (config.json, diffusion_pytorch_model.safetensors, model_index.json, scheduler_config.json).

Uso:   python preparar_modelo.py RUTA_DE_LA_CARPETA_DESCOMPRIMIDA
Ej.:   python preparar_modelo.py "C:\\Users\\yo\\Descargas\\diffusion_pytorch_model"
Los pesos (143 MB) se parten en 2 archivos de <100 MB, porque GitHub rechaza archivos más grandes.
"""
import shutil
import sys
from pathlib import Path

import ddpm_core as core

if len(sys.argv) != 2:
    sys.exit(__doc__)
src = Path(sys.argv[1])
files = {p.name: p.read_bytes() for p in src.iterdir() if p.is_file() and p.suffix in (".json", ".safetensors")}
tmp = Path(__file__).parent / "model" / "_tmp_ordenado"
shutil.rmtree(tmp, ignore_errors=True)
d, missing = core.assemble_model_dir(files, tmp)
if d is None:
    sys.exit(f"Faltan archivos en {src}: {', '.join(missing)}")

from diffusers import UNet2DModel

dest = Path(__file__).parent / "model" / "ddpm-cifar10-32"
shutil.rmtree(dest, ignore_errors=True)
(dest / "scheduler").mkdir(parents=True)
shutil.copy(d / "model_index.json", dest)
shutil.copy(d / "scheduler" / "scheduler_config.json", dest / "scheduler")
UNet2DModel.from_pretrained(d / "unet").save_pretrained(dest / "unet", max_shard_size="80MB", safe_serialization=True)
shutil.rmtree(tmp, ignore_errors=True)
print("Listo ->", dest)
for p in sorted(dest.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(dest)}  {p.stat().st_size / 2**20:.1f} MB")
