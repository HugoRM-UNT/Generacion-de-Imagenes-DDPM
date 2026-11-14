"""
ddpm_core.py - Núcleo de inferencia reproducible para google/ddpm-cifar10-32
=============================================================================
Grupo 10 - Sistemas Inteligentes y Machine Learning (Modelo de difusión DDPM)

Este módulo NO depende de Streamlit. Lo usan: la app (app.py), el script
run_experiments.py y el notebook didáctico.

Decisiones de diseño (justificadas en README.md):
  * El bucle de muestreo está escrito explícitamente (U-Net + scheduler) en vez de
    llamar a DDPMPipeline.__call__, para poder (a) medir tiempos limpios, (b) capturar
    estados intermedios x_t / x0_hat y (c) demostrar comprensión de la arquitectura.
  * El ruido inicial sale SIEMPRE de un torch.Generator("cpu") sembrado, de modo que la
    misma semilla produce el mismo x_T en CPU y en GPU.
  * Ninguna cifra se inventa: todo lo que se reporta se calcula en run_experiments().
"""
from __future__ import annotations

import csv
import importlib.metadata as md
import json
import os
import platform
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Constantes del protocolo experimental (congeladas ANTES de ver resultados)
# --------------------------------------------------------------------------- #
MODEL_ID = "google/ddpm-cifar10-32"
RESOLUTION = 32
STEPS_GRID = (25, 50, 100, 250)                       # Etapa 11
SCHEDULERS = ("DDPM", "DDIM")
CONFIGS = [(s, n) for s in SCHEDULERS for n in STEPS_GRID]
BASELINE = ("DDPM", 100)                              # Exp. C = línea base / referencia
EXPLORATION_SEEDS = [0, 1, 2, 3]                      # solo para pruebas de humo / warm-up
COMPARISON_SEEDS = list(range(100, 116))              # 16 semillas de VALIDACIÓN: comparar y seleccionar (Etapa 13-14)
TEST_SEEDS = list(range(200, 210))                    # 10 semillas de PRUEBA FINAL: no se usan para elegir nada (Etapa 15)
TIMINGS_FIELDS = ["scheduler", "steps", "seed", "seconds", "device",
                  "peak_cuda_mb", "rss_mb", "peak_rss_mb", "timestamp"]


def cfg_key(scheduler: str, steps: int) -> str:
    return f"{scheduler.lower()}_{int(steps)}"


def cfg_label(scheduler: str, steps: int) -> str:
    return f"{scheduler}-{int(steps)}"


# --------------------------------------------------------------------------- #
# Entorno
# --------------------------------------------------------------------------- #
def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
        return True
    except Exception:
        return False


def default_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def env_info(device: str | None = None) -> dict:
    """Etapa 07: versiones, dispositivo, memoria. Todo leído de la máquina real."""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 2**30, 2)
    except Exception:
        info["ram_total_gb"] = None
    versions = {}
    for pkg in ("torch", "diffusers", "huggingface_hub", "numpy", "pandas", "Pillow",
                "scikit-image", "scipy", "matplotlib", "streamlit", "psutil"):
        try:
            versions[pkg] = md.version(pkg)
        except Exception:
            versions[pkg] = None
    info["versions"] = versions
    try:
        import torch
        info["device"] = device or default_device()
        info["torch_threads"] = torch.get_num_threads()
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
    except Exception:
        info["device"] = device or "sin torch"
    return info


def _rss_mb() -> tuple[float | None, float | None]:
    """(RSS actual, pico de RSS del proceso) en MB. El pico es monótono durante la vida del proceso."""
    cur = peak = None
    try:
        import psutil
        cur = psutil.Process().memory_info().rss / 2**20
    except Exception:
        pass
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = ru / (2**20 if platform.system() == "Darwin" else 2**10)
    except Exception:
        pass
    return cur, peak


# --------------------------------------------------------------------------- #
# Carga del modelo y schedulers
# --------------------------------------------------------------------------- #
def local_model_ok(path) -> bool:
    p = Path(path)
    return (p / "model_index.json").exists() and (p / "unet" / "config.json").exists() and (p / "scheduler" / "scheduler_config.json").exists()


def assemble_model_dir(files: dict, dest) -> tuple[Path | None, list[str]]:
    """
    Arma la estructura de Diffusers (model_index.json, unet/, scheduler/) a partir de archivos sueltos
    {nombre: bytes} subidos por el usuario, o de un .zip que los contenga. Devuelve (carpeta | None, faltantes).
    """
    import io
    import zipfile
    flat = {}
    for name, data in files.items():
        base = Path(name).name
        if base.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data if isinstance(data, bytes) else data.read())) as z:
                for zi in z.namelist():
                    if not zi.endswith("/"):
                        flat[Path(zi).name] = z.read(zi)
        else:
            flat[base] = data if isinstance(data, bytes) else data.read()
    dest = Path(dest)
    unet, sched = dest / "unet", dest / "scheduler"
    unet.mkdir(parents=True, exist_ok=True); sched.mkdir(parents=True, exist_ok=True)
    for n, d in flat.items():
        if n == "model_index.json":
            (dest / n).write_bytes(d)
        elif n == "scheduler_config.json":
            (sched / n).write_bytes(d)
        elif n == "config.json" or n.endswith(".safetensors") or n.endswith(".safetensors.index.json"):
            (unet / n).write_bytes(d)
    missing = []
    if not (dest / "model_index.json").exists(): missing.append("model_index.json")
    if not (sched / "scheduler_config.json").exists(): missing.append("scheduler_config.json")
    if not (unet / "config.json").exists(): missing.append("config.json (de la U-Net)")
    if not any(unet.glob("*.safetensors")): missing.append("diffusion_pytorch_model.safetensors")
    return (None if missing else dest), missing


def load_pipeline(model_id: str = MODEL_ID, device: str | None = None,
                  local_files_only: bool = False):
    """Carga el checkpoint oficial compatible con Diffusers y lo congela (eval, sin gradientes)."""
    import torch
    from diffusers import DDPMPipeline
    from diffusers.utils import logging as dlog
    dlog.set_verbosity_error()
    device = device or default_device()
    try:
        pipe = DDPMPipeline.from_pretrained(model_id, local_files_only=local_files_only)
    except (ImportError, RuntimeError) as e:
        if "numpy" in str(e).lower() or "transformers" in str(e).lower():
            raise RuntimeError(
                "Conflicto de versiones en su entorno (un paquete antiguo, normalmente 'transformers', exige numpy<2). "
                "Solución: use un entorno limpio (python -m venv .venv) e instale requirements.txt, o ejecute "
                "'pip install -U transformers peft accelerate'. Detalle original: " + str(e)[:300]) from e
        raise
    pipe.to(device)
    pipe.unet.eval()
    for p in pipe.unet.parameters():
        p.requires_grad_(False)
    pipe.set_progress_bar_config(disable=True)
    pipe._ddpm_device = device
    return pipe


def make_scheduler(base_scheduler, name: str):
    """Construye DDPM o DDIM a partir de la MISMA configuración de entrenamiento del checkpoint."""
    from diffusers import DDIMScheduler, DDPMScheduler
    from diffusers.utils import logging as dlog
    dlog.set_verbosity_error()
    cfg = base_scheduler.config
    if name.upper() == "DDPM":
        return DDPMScheduler.from_config(cfg)
    if name.upper() == "DDIM":
        return DDIMScheduler.from_config(cfg)   # eta = 0 (determinista) en .step()
    raise ValueError(f"Scheduler no soportado: {name}")


def timesteps_for(base_scheduler, name: str, steps: int) -> list[int]:
    s = make_scheduler(base_scheduler, name)
    s.set_timesteps(steps)
    return [int(t) for t in s.timesteps]


def model_summary(pipe) -> dict:
    """Etapas 03/09: parámetros entrenables (reales) y memoria de pesos."""
    unet = pipe.unet
    n_params = sum(p.numel() for p in unet.parameters())
    n_train = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    mem = sum(p.numel() * p.element_size() for p in unet.parameters()) / 2**20
    return {
        "unet_class": unet.__class__.__name__,
        "scheduler_class_in_checkpoint": pipe.scheduler.__class__.__name__,
        "params_total": int(n_params),
        "params_trainable_now": int(n_train),   # 0 porque el modelo está congelado en inferencia
        "weights_mb": round(mem, 2),
        "unet_config": {k: v for k, v in dict(unet.config).items() if not k.startswith("_")},
        "scheduler_config": {k: v for k, v in dict(pipe.scheduler.config).items() if not k.startswith("_")},
    }


def layer_shapes(unet, device: str = "cpu") -> list[dict]:
    """Forma del tensor a la salida de cada bloque principal de la U-Net (con un forward real)."""
    import torch
    rows: list[dict] = []
    hooks = []

    def first_tensor(o):
        if isinstance(o, torch.Tensor):
            return o
        if isinstance(o, (tuple, list)):
            return first_tensor(o[0])
        if hasattr(o, "sample"):
            return o.sample
        return None

    def mk(name):
        def h(_m, _i, o):
            t = first_tensor(o)
            if t is not None:
                rows.append({"bloque": name, "forma_salida": tuple(t.shape[1:]),
                             "parametros": sum(p.numel() for p in _m.parameters())})
        return h

    hooks.append(unet.conv_in.register_forward_hook(mk("conv_in")))
    for i, b in enumerate(unet.down_blocks):
        hooks.append(b.register_forward_hook(mk(f"down_blocks[{i}] {b.__class__.__name__}")))
    hooks.append(unet.mid_block.register_forward_hook(mk(f"mid_block {unet.mid_block.__class__.__name__}")))
    for i, b in enumerate(unet.up_blocks):
        hooks.append(b.register_forward_hook(mk(f"up_blocks[{i}] {b.__class__.__name__}")))
    hooks.append(unet.conv_out.register_forward_hook(mk("conv_out")))
    c, s = unet.config.in_channels, unet.config.sample_size
    with torch.no_grad():
        unet(torch.zeros(1, c, s, s, device=device), torch.tensor([10], device=device))
    for h in hooks:
        h.remove()
    return rows


# --------------------------------------------------------------------------- #
# Muestreo explícito
# --------------------------------------------------------------------------- #
def _to_uint8(x) -> np.ndarray:
    """Tensor (1,3,H,W) en [-1,1] -> uint8 (H,W,3)."""
    x = (x[0].detach().float().cpu() / 2 + 0.5).clamp(0, 1)
    return (x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def _sync(device: str):
    import torch
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def sample_one(unet, base_scheduler, scheduler_name: str, steps: int, seed: int,
               device: str, n_snapshots: int = 0) -> dict:
    """
    Genera UNA imagen 32x32.
      x_T ~ N(0, I) desde la semilla  ->  para t en timesteps: eps = U-Net(x_t, t);
      x_{t'} = scheduler.step(eps, t, x_t)   ->  imagen final.
    Devuelve dict(image=uint8 HxWx3, seconds=float, snapshots=[...]).
    Si n_snapshots > 0 los tiempos incluyen conversiones extra: NO usar para medir latencia.
    """
    import torch
    sched = make_scheduler(base_scheduler, scheduler_name)
    sched.set_timesteps(int(steps), device=device)
    g = torch.Generator("cpu").manual_seed(int(seed))
    shape = (1, unet.config.in_channels, unet.config.sample_size, unet.config.sample_size)
    x = torch.randn(shape, generator=g).to(device)
    ts = sched.timesteps
    snap_idx = set()
    if n_snapshots > 0:
        snap_idx = set(np.unique(np.linspace(0, len(ts) - 1, n_snapshots).round().astype(int)).tolist())
    snaps = []
    _sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, t in enumerate(ts):
            eps = unet(x, t).sample                                   # ε̂_θ(x_t, t)
            out = sched.step(eps, t, x, generator=g)                  # x_{t'} (DDPM: estocástico, DDIM η=0: determinista)
            if i in snap_idx:
                x0 = getattr(out, "pred_original_sample", None)
                snaps.append({"i": int(i), "t": int(t), "xt": _to_uint8(x),
                              "x0": _to_uint8(x0) if x0 is not None else None})
            x = out.prev_sample
        _sync(device)
        dt = time.perf_counter() - t0
    return {"image": _to_uint8(x), "seconds": dt, "snapshots": snaps}


def reproducibility_check(unet, base_scheduler, device: str, seed: int | None = None, steps: int = 25) -> dict:
    """Misma semilla dos veces -> ¿misma imagen?  Otra semilla -> ¿imagen distinta?  (criterio especial del grupo)"""
    seed = COMPARISON_SEEDS[0] if seed is None else seed
    out = {"seed": seed, "steps": steps, "device": device}
    for s in SCHEDULERS:
        a = sample_one(unet, base_scheduler, s, steps, seed, device)["image"]
        b = sample_one(unet, base_scheduler, s, steps, seed, device)["image"]
        c = sample_one(unet, base_scheduler, s, steps, seed + 1, device)["image"]
        out[s] = {"identical_same_seed": bool(np.array_equal(a, b)),
                  "max_abs_pixel_diff_same_seed": int(np.abs(a.astype(int) - b.astype(int)).max()),
                  "differs_with_other_seed": bool(not np.array_equal(a, c))}
    return out


def run_final_test(out_dir, selected: tuple[str, int], model_id: str = MODEL_ID, device: str | None = None,
                   local_files_only: bool = False, progress=None, log=print) -> Path:
    """Etapa 15: evalúa UNA vez la configuración elegida y la línea base en semillas de prueba nunca usadas."""
    cfgs = [BASELINE] if tuple(selected) == BASELINE else [BASELINE, tuple(selected)]
    return run_experiments(Path(out_dir) / "final_test", model_id, seeds=TEST_SEEDS, configs=cfgs, device=device,
                           local_files_only=local_files_only, progress=progress, log=log, n_trajectory_snapshots=0)


# --------------------------------------------------------------------------- #
# Experimentos (Etapas 08, 11, 12, 13)
# --------------------------------------------------------------------------- #
def save_png(arr: np.ndarray, path: Path):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def load_png(path: Path) -> np.ndarray | None:
    from PIL import Image
    path = Path(path)
    return np.array(Image.open(path).convert("RGB")) if path.exists() else None


def image_path(out_dir, scheduler: str, steps: int, seed: int) -> Path:
    return Path(out_dir) / "images" / cfg_key(scheduler, steps) / f"seed_{int(seed)}.png"


def _revision_of(model_id: str) -> str | None:
    """Hash de commit del checkpoint en caché (trazabilidad, Etapa 03)."""
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(model_id, local_files_only=True)).name
    except Exception:
        return None


def _save_model_card(model_id: str, out_dir: Path) -> dict:
    info = {}
    try:
        from huggingface_hub import ModelCard
        card = ModelCard.load(model_id)
        (out_dir / "model_card.md").write_text(str(card.content), encoding="utf-8")
        info["license"] = card.data.to_dict().get("license")
        info["card_saved"] = True
    except Exception as e:  # sin red o repo local
        info["card_saved"] = False
        info["card_error"] = repr(e)[:200]
    return info


def run_experiments(out_dir="results", model_id: str = MODEL_ID, seeds=None, configs=None,
                    device: str | None = None, local_files_only: bool = False,
                    n_trajectory_snapshots: int = 8, progress=None, log=print) -> Path:
    """
    Ejecuta todas las configuraciones sobre las MISMAS semillas, guarda imágenes, tiempos,
    memoria, trayectorias de denoising y métricas. Es reanudable: salta lo ya calculado.
    progress(frac, texto) es opcional (lo usa la barra de Streamlit).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(COMPARISON_SEEDS if seeds is None else seeds)
    configs = list(CONFIGS if configs is None else configs)
    device = device or default_device()

    log(f"[1/5] Cargando {model_id} en {device} ...")
    pipe = load_pipeline(model_id, device, local_files_only)
    unet, base = pipe.unet, pipe.scheduler

    env = env_info(device)
    summ = model_summary(pipe)
    summ["revision"] = _revision_of(model_id)
    summ["model_id"] = model_id
    summ.update(_save_model_card(model_id, out_dir))
    (out_dir / "env.json").write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "model_info.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    try:
        (out_dir / "layer_shapes.json").write_text(
            json.dumps(layer_shapes(unet, device), indent=2, default=list), encoding="utf-8")
    except Exception as e:
        log(f"   (no se pudieron registrar formas por bloque: {e!r})")
    run_cfg = {"model_id": model_id, "baseline": list(BASELINE), "configs": [list(c) for c in configs],
               "comparison_seeds": seeds, "exploration_seeds": EXPLORATION_SEEDS,
               "resolution": RESOLUTION, "batch_size_timing": 1, "ddim_eta": 0.0,
               "started": datetime.now().isoformat(timespec="seconds"),
               "notes": "Semillas de comparación reservadas; modelo congelado; no se ajusta nada con ellas."}
    (out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    try:
        rep = reproducibility_check(unet, base, device)
        (out_dir / "reproducibility.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        log("   Reproducibilidad:", {k: v for k, v in rep.items() if k in SCHEDULERS})
    except Exception as e:
        log(f"   (comprobación de reproducibilidad omitida: {e!r})")

    tpath = out_dir / "timings.csv"
    done = set()
    if tpath.exists():
        import pandas as pd
        d = pd.read_csv(tpath)
        done = {(r.scheduler, int(r.steps), int(r.seed)) for r in d.itertuples()}
    new_file = not tpath.exists()
    f = open(tpath, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=TIMINGS_FIELDS)
    if new_file:
        w.writeheader()

    total = len(configs) * len(seeds)
    k = 0
    import torch
    log(f"[2/5] Generando {len(configs)} configuraciones x {len(seeds)} semillas = {total} imágenes")
    for sname, steps in configs:
        pending = [s for s in seeds if (sname, steps, s) not in done or not image_path(out_dir, sname, steps, s).exists()]
        if pending:   # warm-up (no se registra): evita contar la inicialización de kernels/caché
            sample_one(unet, base, sname, steps, EXPLORATION_SEEDS[0], device)
        for seed in seeds:
            k += 1
            if seed not in pending:
                continue
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            r = sample_one(unet, base, sname, steps, seed, device)
            rss, peak_rss = _rss_mb()
            save_png(r["image"], image_path(out_dir, sname, steps, seed))
            w.writerow({"scheduler": sname, "steps": steps, "seed": seed,
                        "seconds": round(r["seconds"], 5), "device": device,
                        "peak_cuda_mb": round(torch.cuda.max_memory_allocated() / 2**20, 2) if device.startswith("cuda") else "",
                        "rss_mb": round(rss, 1) if rss else "", "peak_rss_mb": round(peak_rss, 1) if peak_rss else "",
                        "timestamp": datetime.now().isoformat(timespec="seconds")})
            f.flush()
            msg = f"{cfg_label(sname, steps)} seed={seed}: {r['seconds']:.3f}s"
            log("   " + msg)
            if progress:
                progress(k / total, msg)
    f.close()

    log("[3/5] Trayectorias de denoising (una semilla de comparación por configuración)")
    tdir = out_dir / "trajectories"
    tdir.mkdir(exist_ok=True)
    for sname, steps in (configs if n_trajectory_snapshots > 0 else []):
        fp = tdir / f"{cfg_key(sname, steps)}_seed{seeds[0]}.npz"
        if fp.exists():
            continue
        r = sample_one(unet, base, sname, steps, seeds[0], device, n_snapshots=n_trajectory_snapshots)
        np.savez_compressed(fp, t=[s["t"] for s in r["snapshots"]], i=[s["i"] for s in r["snapshots"]],
                            xt=np.stack([s["xt"] for s in r["snapshots"]]),
                            x0=np.stack([s["x0"] for s in r["snapshots"]]), final=r["image"])

    log("[4/5] Calculando métricas")
    compute_metrics(out_dir)
    log("[5/5] Listo ->", out_dir.resolve())
    return out_dir


# --------------------------------------------------------------------------- #
# Métricas (todas calculadas desde las imágenes reales guardadas)
# --------------------------------------------------------------------------- #
def _tv(img01: np.ndarray) -> float:
    """Variación total media por píxel: indicador de ruido residual / alta frecuencia."""
    dx = np.abs(np.diff(img01, axis=1)).mean()
    dy = np.abs(np.diff(img01, axis=0)).mean()
    return float(dx + dy)


def compute_metrics(out_dir) -> None:
    import pandas as pd
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    out_dir = Path(out_dir)
    tpath = out_dir / "timings.csv"
    if not tpath.exists():
        return
    t = pd.read_csv(tpath)
    t = t.drop_duplicates(["scheduler", "steps", "seed"], keep="last")
    cfgs = sorted({(r.scheduler, int(r.steps)) for r in t.itertuples()}, key=lambda c: (c[0], c[1]))
    seeds = sorted(t.seed.unique())
    imgs = {}
    for s, n in cfgs:
        for sd in seeds:
            a = load_png(image_path(out_dir, s, n, sd))
            if a is not None:
                imgs[(s, n, sd)] = a.astype(np.float32) / 255.0

    rows = []
    for (s, n, sd), a in imgs.items():
        row = {"scheduler": s, "steps": n, "seed": sd,
               "tv": _tv(a), "contrast": float(a.std())}
        b = imgs.get((*BASELINE, sd))
        if b is not None:
            mse = float(((a - b) ** 2).mean())
            row["mse_vs_baseline"] = mse
            row["psnr_vs_baseline"] = float(psnr(b, a, data_range=1.0)) if mse > 0 else float("nan")
            row["ssim_vs_baseline"] = float(ssim(b, a, channel_axis=2, data_range=1.0))
        rows.append(row)
    per_img = pd.DataFrame(rows)
    per_img.to_csv(out_dir / "metrics_per_image.csv", index=False)

    try:
        from scipy import stats
        tcrit = lambda n: stats.t.ppf(0.975, max(n - 1, 1))
    except Exception:
        tcrit = lambda n: 1.96
    summ = []
    for s, n in cfgs:
        ts = t[(t.scheduler == s) & (t.steps == n)]
        sec = ts.seconds.values
        stack = np.stack([imgs[(s, n, sd)] for sd in seeds if (s, n, sd) in imgs]) if any((s, n, sd) in imgs for sd in seeds) else None
        d = {"scheduler": s, "steps": n, "n_images": len(sec),
             "sec_mean": sec.mean(), "sec_std": sec.std(ddof=1) if len(sec) > 1 else 0.0,
             "sec_median": float(np.median(sec)),
             "sec_ci95": tcrit(len(sec)) * sec.std(ddof=1) / np.sqrt(len(sec)) if len(sec) > 1 else 0.0,
             "sec_per_step_ms": 1000 * sec.mean() / n,
             "peak_cuda_mb": pd.to_numeric(ts.peak_cuda_mb, errors="coerce").max(),
             "rss_mb_max": pd.to_numeric(ts.rss_mb, errors="coerce").max()}
        if stack is not None and len(stack) > 1:
            flat = stack.reshape(len(stack), -1)
            d["div_pairwise_rms"] = float(np.mean([np.sqrt(((flat[i] - flat[j]) ** 2).mean())
                                                   for i, j in combinations(range(len(flat)), 2)]))
            d["div_pixel_std"] = float(stack.std(axis=0).mean())
        pi = per_img[(per_img.scheduler == s) & (per_img.steps == n)]
        for col in ("tv", "contrast", "psnr_vs_baseline", "ssim_vs_baseline"):
            if col in pi:
                d[col + "_mean"] = float(pi[col].replace([np.inf, -np.inf], np.nan).mean())
        summ.append(d)
    pd.DataFrame(summ).to_csv(out_dir / "summary.csv", index=False)


# --------------------------------------------------------------------------- #
# Selección (Etapa 14): criterio declarado de antemano
# --------------------------------------------------------------------------- #
def select_config(summary, ratings, tol: float = 0.5):
    """
    Regla (pre-declarada): entre las configuraciones cuya calificación media de evaluadores
    sea >= (mejor media - tol), elegir la de menor tiempo medio por imagen.
    Requiere calificaciones humanas; si no existen devuelve None (no se inventa calidad).
    """
    if ratings is None or len(ratings) == 0:
        return None
    r = ratings.groupby(["scheduler", "steps"]).score.agg(["mean", "count"]).reset_index()
    m = summary.merge(r, on=["scheduler", "steps"], how="inner")
    if m.empty:
        return None
    ok = m[m["mean"] >= m["mean"].max() - tol]
    best = ok.sort_values("sec_mean").iloc[0]
    return {"scheduler": best.scheduler, "steps": int(best.steps), "rating_mean": float(best["mean"]),
            "sec_mean": float(best.sec_mean), "candidates": ok[["scheduler", "steps"]].values.tolist(),
            "tol": tol}


# --------------------------------------------------------------------------- #
# Utilidades de visualización
# --------------------------------------------------------------------------- #
def make_grid(images: list[np.ndarray | None], ncols: int, scale: int = 4, pad: int = 2) -> np.ndarray:
    """Mosaico de imágenes uint8 HxWx3 (None -> hueco gris), ampliado con vecino más cercano."""
    from PIL import Image
    h = w = RESOLUTION * scale
    nrows = int(np.ceil(len(images) / ncols))
    canvas = np.full((nrows * (h + pad) + pad, ncols * (w + pad) + pad, 3), 245, np.uint8)
    for k, im in enumerate(images):
        r, c = divmod(k, ncols)
        tile = np.full((h, w, 3), 200, np.uint8) if im is None else np.array(
            Image.fromarray(im).resize((w, h), Image.NEAREST))
        canvas[pad + r * (h + pad): pad + r * (h + pad) + h, pad + c * (w + pad): pad + c * (w + pad) + w] = tile
    return canvas
