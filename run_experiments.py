#!/usr/bin/env python
"""
Ejecuta TODO el protocolo experimental (Etapas 08-15) sobre google/ddpm-cifar10-32.

  python run_experiments.py                       # protocolo completo: 8 configs x 16 semillas
  python run_experiments.py --seeds 10            # 10 semillas (mínimo exigido por la guía)
  python run_experiments.py --device cpu          # forzar CPU
  python run_experiments.py --model-path ./ddpm-cifar10-32 --offline   # checkpoint descargado a mano

Es reanudable: si se interrumpe, vuelva a ejecutarlo y continúa donde quedó.
"""
import argparse

import ddpm_core as core

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results")
ap.add_argument("--model-path", default=core.MODEL_ID, help="ID de Hugging Face o carpeta local")
ap.add_argument("--device", default=None, help="cpu | cuda (por defecto autodetecta)")
ap.add_argument("--seeds", type=int, default=len(core.COMPARISON_SEEDS), help="cantidad de semillas (10-20)")
ap.add_argument("--offline", action="store_true", help="no consultar Hugging Face (checkpoint ya en caché/local)")
a = ap.parse_args()

seeds = core.COMPARISON_SEEDS[:a.seeds] if a.seeds <= len(core.COMPARISON_SEEDS) else list(range(100, 100 + a.seeds))
core.run_experiments(a.out, a.model_path, seeds=seeds, device=a.device, local_files_only=a.offline)
