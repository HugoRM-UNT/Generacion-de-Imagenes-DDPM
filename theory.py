"""
theory.py - Matemática de la difusión (solo NumPy/Matplotlib; funciona sin torch ni pesos).
Calendario lineal de Ho et al. (2020): beta_1=1e-4 ... beta_T=0.02, T=1000, que es el
usado por el scheduler del checkpoint google/ddpm-cifar10-32 (se verifica con scheduler_config
cuando el modelo está cargado).
"""
from __future__ import annotations

import numpy as np

T_TRAIN = 1000


def linear_betas(T: int = T_TRAIN, b0: float = 1e-4, b1: float = 0.02) -> np.ndarray:
    return np.linspace(b0, b1, T, dtype=np.float64)


def alpha_bars(betas: np.ndarray) -> np.ndarray:
    return np.cumprod(1.0 - betas)


def q_sample(x0: np.ndarray, t: int, eps: np.ndarray, abar: np.ndarray) -> np.ndarray:
    """x_t = sqrt(abar_t) x0 + sqrt(1-abar_t) eps   (proceso directo en forma cerrada)."""
    return np.sqrt(abar[t]) * x0 + np.sqrt(1.0 - abar[t]) * eps


def snr(abar: np.ndarray) -> np.ndarray:
    return abar / (1.0 - abar)


def strided_timesteps(steps: int, T: int = T_TRAIN) -> list[int]:
    """Espaciado 'leading' de diffusers: (0..n-1)*(T//n) en orden descendente."""
    ratio = T // steps
    return [int(v) for v in (np.arange(steps) * ratio).round()[::-1]]


def demo_image(size: int = 32) -> np.ndarray:
    """Imagen sintética ORIGINAL (cielo, sol, colinas) en [-1,1], HxWx3, para ilustrar la difusión."""
    y, x = np.mgrid[0:size, 0:size] / (size - 1)
    sky = np.stack([0.35 + 0.4 * y, 0.55 + 0.3 * y, 0.95 - 0.2 * y], -1)
    img = sky.copy()
    sun = (x - 0.72) ** 2 + (y - 0.25) ** 2 < 0.012
    img[sun] = [1.0, 0.85, 0.2]
    hill = y > 0.62 + 0.08 * np.sin(2 * np.pi * (x * 1.3 + 0.1))
    img[hill] = [0.15, 0.55 + 0.2 * (1 - y[hill][:, None].mean()), 0.2]
    hill2 = y > 0.8 + 0.05 * np.sin(2 * np.pi * (x * 2.1 + 0.4))
    img[hill2] = [0.08, 0.35, 0.12]
    return (img * 2 - 1).clip(-1, 1)


def to_img01(x: np.ndarray) -> np.ndarray:
    return ((x / 2 + 0.5).clip(0, 1) * 255).round().astype(np.uint8)


def iterative_vs_closed_form(x0_pixel: float = 0.6, t: int = 500, n: int = 20000, seed: int = 0):
    """
    Comprueba numéricamente que aplicar t veces q(x_k | x_{k-1}) = N(sqrt(1-b_k) x_{k-1}, b_k I)
    equivale a la forma cerrada. Devuelve (media_iter, std_iter, media_cf, std_cf).
    """
    rng = np.random.default_rng(seed)
    betas = linear_betas()
    ab = alpha_bars(betas)
    x = np.full(n, x0_pixel)
    for k in range(t + 1):
        x = np.sqrt(1 - betas[k]) * x + np.sqrt(betas[k]) * rng.standard_normal(n)
    cf = np.sqrt(ab[t]) * x0_pixel + np.sqrt(1 - ab[t]) * rng.standard_normal(n)
    return x.mean(), x.std(), cf.mean(), cf.std()


# --------------------------------------------------------------------------- #
# Diagrama ORIGINAL de la arquitectura (Etapa 09 / sección 4 de la guía)
# --------------------------------------------------------------------------- #
REFERENCE_UNET_CONFIG = {
    "sample_size": 32, "in_channels": 3, "out_channels": 3, "layers_per_block": 2,
    "block_out_channels": [128, 256, 256, 256],
    "down_block_types": ["DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "DownBlock2D"],
    "up_block_types": ["UpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D"],
    "norm_num_groups": 32, "act_fn": "silu", "time_embedding_type": "positional",
}


def unet_diagram(cfg: dict | None = None):
    """Dibuja la U-Net (ε-predictor) con dimensiones de tensor derivadas de la configuración."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    cfg = cfg or REFERENCE_UNET_CONFIG
    chans = list(cfg["block_out_channels"])
    nb = len(chans)
    res0 = cfg["sample_size"]
    L = cfg["layers_per_block"]
    c_in, c_out = cfg["in_channels"], cfg["out_channels"]

    fig, ax = plt.subplots(figsize=(13, 7.6))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 8.3)
    ax.axis("off")
    C = {"down": "#dbeafe", "up": "#dcfce7", "mid": "#fef3c7", "io": "#f3f4f6", "t": "#fce7f3", "edge": "#334155"}

    def box(x, y, w, h, txt, fc, fs=8.2, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=fc, ec=C["edge"], lw=1))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color="#0f172a")

    def arrow(p, q, style="-|>", ls="-", color="#334155", rad=0.0, lw=1.1):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=10, lw=lw, ls=ls,
                                     color=color, connectionstyle=f"arc3,rad={rad}"))

    ys = np.linspace(5.3, 2.5, nb)
    xl, xr, bw, bh = 0.9, 9.0, 3.1, 0.82
    res = res0
    res_at = []
    for i in range(nb):
        res_at.append(res)
        res = res // 2 if i < nb - 1 else res
    top_in, top_conv = 7.35, 6.5
    box(xl, top_in, bw, 0.62, f"Entrada x_t\n{c_in}×{res0}×{res0}", C["io"], bold=True)
    box(xl, top_conv, bw, 0.42, f"conv_in → {chans[0]}×{res0}×{res0}", C["io"], 7.6)
    arrow((xl + bw / 2, top_in), (xl + bw / 2, top_conv + 0.42))
    arrow((xl + bw / 2, top_conv), (xl + bw / 2, ys[0] + bh))
    for i in range(nb):
        typ = cfg["down_block_types"][i]
        attn = "Attn" in typ
        r = res_at[i]
        rout = r // 2 if i < nb - 1 else r
        txt = (f"Down {i}: {typ}\n{L}×ResNet{' + Self-Attention' if attn else ''}  ({chans[i]} canales)\n"
               f"{r}×{r}" + (f" → {rout}×{rout} (↓)" if i < nb - 1 else " (sin ↓)"))
        box(xl, ys[i], bw, bh, txt, C["down"])
        if i < nb - 1:
            arrow((xl + bw / 2, ys[i]), (xl + bw / 2, ys[i + 1] + bh))
    midy, midw = 0.95, 8.3
    box(xl + 1.2, midy, midw, 0.9,
        f"Bloque medio (UNetMidBlock2D): ResNet → Self-Attention → ResNet   ·   {chans[-1]} canales  ·  {res_at[-1]}×{res_at[-1]}",
        C["mid"], 8.2, True)
    arrow((xl + bw / 2, ys[-1]), (xl + bw / 2, midy + 0.9))
    for k in range(nb):
        i = nb - 1 - k
        typ = cfg["up_block_types"][k]
        attn = "Attn" in typ
        r = res_at[i]
        rout = r * 2 if k < nb - 1 else r
        cch = list(reversed(chans))[k]
        txt = (f"Up {k}: {typ}\n{L + 1}×ResNet{' + Self-Attention' if attn else ''}  ({cch} canales)\n"
               f"{r}×{r}" + (f" → {rout}×{rout} (↑)" if k < nb - 1 else " (sin ↑)"))
        box(xr, ys[i], bw, bh, txt, C["up"])
        arrow((xl + bw, ys[i] + bh / 2), (xr, ys[i] + bh / 2), ls="--", color="#2563eb", lw=1.0)
        ax.text((xl + bw + xr) / 2, ys[i] + bh / 2 + 0.12, "skip: concatenación", fontsize=7, ha="center", color="#2563eb")
        if k > 0:
            arrow((xr + bw / 2, ys[i + 1] + bh), (xr + bw / 2, ys[i]))
    arrow((xl + 1.2 + midw, midy + 0.45), (xr + bw / 2, ys[nb - 1]), rad=-0.3)
    box(xr, top_conv, bw, 0.42, f"GroupNorm+SiLU+conv_out → {c_out}×{res0}×{res0}", C["io"], 7.2)
    arrow((xr + bw / 2, ys[0] + bh), (xr + bw / 2, top_conv))
    box(xr, top_in, bw, 0.62, f"Salida ε̂_θ(x_t, t)\n{c_out}×{res0}×{res0}  (ruido estimado)", C["io"], bold=True)
    arrow((xr + bw / 2, top_conv + 0.42), (xr + bw / 2, top_in))
    # Embedding temporal
    box(4.9, 6.35, 3.6, 1.6,
        "Timestep t ∈ {0..999}\n↓ embedding sinusoidal\n↓ MLP (Linear→SiLU→Linear)\nvector de tiempo (4×C₀)\n↳ se SUMA en cada ResNet",
        C["t"], 7.6)
    arrow((4.9, 7.1), (xl + bw, 5.9), ls=":", color="#be185d")
    arrow((8.5, 7.1), (xr, 5.9), ls=":", color="#be185d")
    ax.text(6.6, 0.4, "Flujo de aprendizaje (entrenamiento): L = E‖ε − ε̂_θ(√ᾱ_t x₀ + √(1−ᾱ_t) ε, t)‖²  → retropropagación sobre TODOS los pesos de la U‑Net.\n"
            "En inferencia los pesos están congelados: el scheduler usa ε̂ para pasar de x_t a x_{t′} y repite hasta x₀.",
            ha="center", va="center", fontsize=8.2, color="#0f172a")
    ax.set_title("U‑Net estimadora de ruido del DDPM (diagrama original; dimensiones por muestra, sin batch)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig
