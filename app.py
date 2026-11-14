"""
Aplicativo Streamlit - Grupo 10 - Evaluación de generación de imágenes con un DDPM preentrenado
Modelo: google/ddpm-cifar10-32 (incondicional, 32x32).  Ejecutar:  streamlit run app.py
"""
from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import content
import ddpm_core as core
import theory

APP_DIR = Path(__file__).parent
IS_CLOUD = Path("/mount/src").exists() or os.environ.get("STREAMLIT_SHARING_MODE") is not None
st.set_page_config(page_title="DDPM CIFAR-10 · Grupo 10", page_icon="🌫️", layout="wide")

# ----------------------------------------------------------------------------- utilidades
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
.badge {display:inline-block; padding:2px 10px; border-radius:12px; font-size:0.78rem; font-weight:600;}
.ok {background:#dcfce7; color:#166534;} .pend {background:#fef3c7; color:#92400e;}
.small {font-size:0.85rem; color:#64748b;}
</style>""", unsafe_allow_html=True)

LOCAL_MODEL = APP_DIR / "model" / "ddpm-cifar10-32"
HAS_LOCAL = core.local_model_ok(LOCAL_MODEL)
DEFAULT_MODEL = str(LOCAL_MODEL) if HAS_LOCAL else core.MODEL_ID
default_res = os.environ.get("DDPM_RESULTS_DIR", str(APP_DIR / "results"))
st.sidebar.title("Panel de control")
RES = Path(st.sidebar.text_input("Carpeta de resultados", default_res,
                                 help="Aquí lee la app las imágenes, tiempos y métricas generados por run_experiments.py"))
FINAL = RES / "final_test"


def mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner=False)
def _csv(path: str, mt: float):
    return pd.read_csv(path)


def load_csv(name: str, base: Path | None = None):
    p = (base or RES) / name
    return _csv(str(p), mtime(p)) if p.exists() else None


def load_json(name: str, base: Path | None = None):
    p = (base or RES) / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def get_img(s, n, seed, base: Path | None = None):
    return core.load_png(core.image_path(base or RES, s, n, seed))


def badge(ok: bool, yes="evidencia disponible", no="pendiente: ejecutar experimentos") -> str:
    return f'<span class="badge {"ok" if ok else "pend"}">{yes if ok else no}</span>'


def show(img, width=None, caption=None):
    st.image(img, width=width, caption=caption)


def run_ready() -> bool:
    return (RES / "timings.csv").exists()


summary = load_csv("summary.csv")
timings = load_csv("timings.csv")
per_img = load_csv("metrics_per_image.csv")
model_info = load_json("model_info.json")
env = load_json("env.json")

with st.sidebar:
    st.markdown("**Modelo**  \n`google/ddpm-cifar10-32`")
    if model_info and model_info.get("revision"):
        st.caption(f"Revisión: `{model_info['revision'][:10]}…`")
    st.markdown("**Estado de resultados** " + badge(run_ready(), "hay resultados", "sin resultados"), unsafe_allow_html=True)
    if run_ready():
        st.caption(f"{len(timings)} imágenes registradas · dispositivo: {timings.device.iloc[0]}")
    else:
        st.info("Aún no hay resultados reales. Ejecute `python run_experiments.py` o use la pestaña **Ejecutar experimentos**.")
    st.markdown("---")
    st.caption("Esta app **no inventa cifras**: toda tabla o gráfico sale de los archivos de `results/` o de una ejecución en vivo.")

st.title("Evaluación de generación de imágenes con un DDPM preentrenado")
st.caption("Grupo 10 · Sistemas Inteligentes y Machine Learning · Modelo de difusión (DDPM) · google/ddpm-cifar10-32")

tabs = st.tabs(["Resumen", "Teoría y arquitectura", "Difusión directa", "Galería y demo", "Evaluación",
                "Calificación a ciegas", "Generación en vivo", "Ejecutar experimentos", "16 etapas",
                "Limitaciones y sustentación"])



def model_source_ui(key: str):
    """Elige el origen del modelo. Prioridad: carpeta local del proyecto; si no está, pide subirlo o descargarlo."""
    import tempfile
    opts = (["Modelo local (carpeta model/ del proyecto)"] if HAS_LOCAL else []) + \
           ["Subir archivos del modelo", "Descargar de Hugging Face", "Otra carpeta o ID"]
    choice = st.radio("Origen del modelo", opts, key=f"src_{key}", horizontal=True)
    if not HAS_LOCAL:
        st.info("No se encontró el modelo en la carpeta `model/ddpm-cifar10-32/` del proyecto: súbalo aquí, o descárguelo de Hugging Face.")
    if choice.startswith("Modelo local"):
        return str(LOCAL_MODEL), True
    if choice == "Descargar de Hugging Face":
        return core.MODEL_ID, False
    if choice == "Otra carpeta o ID":
        p = st.text_input("Carpeta local o ID de Hugging Face", core.MODEL_ID, key=f"txt_{key}")
        return p, Path(p).exists()
    st.caption("Suba **model_index.json**, **scheduler_config.json**, **config.json** y **diffusion_pytorch_model.safetensors** "
               "(los archivos tal como vienen, juntos), o un único .zip que los contenga. Si la carpeta `unet` está en fragmentos, suba también el `.index.json`.")
    ups = st.file_uploader("Archivos del modelo", accept_multiple_files=True, type=["json", "safetensors", "zip"], key=f"up_{key}")
    if not ups:
        return None, True
    sig = tuple(sorted((u.name, u.size) for u in ups))
    dest = Path(tempfile.gettempdir()) / "ddpm_modelo_subido"
    if st.session_state.get("up_sig") != sig:
        import shutil
        shutil.rmtree(dest, ignore_errors=True)
        path, missing = core.assemble_model_dir({u.name: u.getvalue() for u in ups}, dest)
        if path is None:
            st.error("Faltan archivos: " + ", ".join(missing)); return None, True
        st.session_state["up_sig"] = sig
    st.success("Modelo recibido y ordenado correctamente.")
    return str(dest), True


# ============================================================================= 1. RESUMEN
with tabs[0]:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Situación problemática y propósito")
        st.write("Un equipo de investigación quiere entender cómo un **modelo de difusión construye imágenes a partir de ruido** "
                 "y cómo las decisiones de muestreo (número de pasos y programador) afectan **tiempo, memoria y calidad observable**. "
                 "No se entrena nada: se analiza el DDPM público preentrenado `google/ddpm-cifar10-32`.")
        st.markdown("**Objetivo general.** Implementar inferencia y comparar configuraciones de muestreo en un DDPM CIFAR-10 preentrenado.")
        st.markdown("""**Objetivos específicos**
- Explicar difusión directa, estimación del ruido y generación inversa (pestañas *Teoría* y *Difusión directa*).
- Usar el modelo oficial compatible con Diffusers (`ddpm_core.py`).
- Comparar pasos {25, 50, 100, 250} y programadores DDPM/DDIM con **semillas fijadas** (*Galería* y *Evaluación*).
- Analizar diversidad, calidad observable y costo de cómputo **sin confundirlos con exactitud de clasificación**.""")
    with c2:
        st.subheader("Cómo usar esta app")
        st.markdown("""
- **Qué admite:** semillas enteras, número de pasos (1–1000) y scheduler (DDPM o DDIM). **No admite texto ni imágenes de entrada** para generar (modelo incondicional). En *Difusión directa* sí puede subir una imagen, solo para ilustrar el ruido.
- **Qué modelo carga:** `google/ddpm-cifar10-32` (U-Net, ≈35.7 M parámetros), 32×32 RGB.
- **Cómo leer la salida:** cada imagen es **una muestra aleatoria** de lo aprendido de CIFAR-10; no representa un objeto pedido por el usuario. Misma semilla + misma configuración ⇒ misma imagen.
- **Sin GPU:** use la **galería precalculada** (carpeta `results/`). La generación en vivo funciona en CPU, solo es más lenta.
""")
    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Configuraciones", f"{len(core.CONFIGS)}", "DDPM/DDIM × {25,50,100,250}")
    k2.metric("Semillas de validación", f"{len(core.COMPARISON_SEEDS)}", "100–115")
    k3.metric("Semillas de prueba final", f"{len(core.TEST_SEEDS)}", "200–209")
    k4.metric("Imágenes generadas", f"{len(timings) if timings is not None else 0}")
    st.caption("Los contadores reflejan lo realmente ejecutado en la carpeta de resultados seleccionada.")


# ============================================================================= 2. TEORÍA
with tabs[1]:
    st.header("Fundamentos teóricos y arquitectura")
    st.subheader("1. Idea general")
    st.write("Un modelo de difusión aprende a **invertir un proceso que destruye los datos con ruido**. Generar = partir de ruido puro "
             "x_T ∼ N(0, I) y **eliminar ruido paso a paso** hasta obtener x₀ (una imagen nueva). La red no genera la imagen de golpe: "
             "en cada paso solo estima el ruido presente; el *scheduler* usa esa estimación para dar un pequeño paso hacia una imagen más limpia.")

    st.subheader("2. Difusión directa (fija, sin parámetros)")
    st.latex(r"q(x_t\mid x_{t-1})=\mathcal N\!\big(\sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\big),\qquad \alpha_t=1-\beta_t,\ \ \bar\alpha_t=\prod_{s=1}^{t}\alpha_s")
    st.latex(r"\boxed{x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,\qquad \varepsilon\sim\mathcal N(0,I)}")
    st.write("Calendario lineal: β₁ = 10⁻⁴ … β_T = 0.02, T = 1000. ᾱ_t es la **proporción de señal preservada**; ᾱ_T ≈ 0 ⇒ x_T es ruido casi puro.")

    st.subheader("3. Entrenamiento (ya hecho por los autores del checkpoint)")
    st.latex(r"\mathcal L(\theta)=\mathbb E_{x_0,\,t,\,\varepsilon}\Big[\big\|\varepsilon-\varepsilon_\theta\big(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,\ t\big)\big\|^2\Big]")
    st.write("Se elige una imagen x₀, un t aleatorio y un ruido ε; se forma x_t y la U-Net aprende a **predecir ε** (retropropagación sobre todos sus pesos). "
             "Esta es la fase que **no repetimos**.")

    st.subheader("4. Inferencia: papel del scheduler")
    cL, cR = st.columns(2)
    with cL:
        st.markdown("**Estimación de la imagen limpia** (común):")
        st.latex(r"\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\varepsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}")
        st.markdown("**DDPM** (estocástico, con t → t′ saltando pasos):")
        st.latex(r"x_{t'}=\underbrace{\tfrac{\sqrt{\bar\alpha_{t'}}\,\beta^{ef}_t}{1-\bar\alpha_t}\hat x_0+\tfrac{\sqrt{\alpha^{ef}_t}(1-\bar\alpha_{t'})}{1-\bar\alpha_t}x_t}_{\tilde\mu_t}+\sigma_t z,\quad z\sim\mathcal N(0,I)")
        st.caption("α^ef_t = ᾱ_t/ᾱ_t′, β^ef_t = 1 − α^ef_t. Con pasos consecutivos se recupera el algoritmo 2 de Ho et al. (2020). σ_t depende del `variance_type` del scheduler.")
    with cR:
        st.markdown("**DDIM** (Song et al., 2021), mismo modelo entrenado:")
        st.latex(r"x_{t'}=\sqrt{\bar\alpha_{t'}}\,\hat x_0+\sqrt{1-\bar\alpha_{t'}-\sigma_t^2}\;\varepsilon_\theta(x_t,t)+\sigma_t z")
        st.latex(r"\sigma_t=\eta\sqrt{\tfrac{1-\bar\alpha_{t'}}{1-\bar\alpha_t}}\sqrt{1-\tfrac{\bar\alpha_t}{\bar\alpha_{t'}}}")
        st.caption("Con η = 0, σ_t = 0: muestreo **determinista** (x_T determina la imagen). Es el valor usado aquí.")
    st.markdown("**Pasos y timesteps visitados.** El scheduler elige un subconjunto de los 1000 pasos de entrenamiento:")
    ns = st.select_slider("Número de pasos de inferencia", options=[10, 25, 50, 100, 250, 500, 1000], value=25, key="ns_theory")
    ts = theory.strided_timesteps(ns)
    ab = theory.alpha_bars(theory.linear_betas())
    df_ts = pd.DataFrame({"orden": range(1, len(ts) + 1), "timestep t": ts, "ᾱ_t": [round(float(ab[t]), 5) for t in ts]})
    c1, c2 = st.columns([1, 2])
    c1.dataframe(df_ts.head(8), hide_index=True)
    c1.caption(f"Primeros 8 de {ns} (espaciado `leading`: cada {1000 // ns} pasos).")
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    ax.plot(range(1000), ab, color="#94a3b8", label="ᾱ_t (señal preservada)")
    ax.scatter(ts, ab[ts], s=10, color="#2563eb", label=f"timesteps visitados ({ns})", zorder=3)
    ax.set_xlabel("t"); ax.set_ylabel("ᾱ_t"); ax.legend(fontsize=8); ax.grid(alpha=.2)
    c2.pyplot(fig); plt.close(fig)

    st.subheader("5. Arquitectura: U-Net estimadora de ruido")
    fig = theory.unet_diagram(model_info["unet_config"] if model_info and "unet_config" in model_info else None)
    st.pyplot(fig); plt.close(fig)
    st.markdown("""
| Elemento | Detalle |
|---|---|
| **Entradas** | x_t: tensor 3×32×32 (por imagen) · t: entero ∈ [0, 999] |
| **Dimensión de tensores** | 3×32×32 → 128×32×32 → 128×16×16 → 256×8×8 → 256×4×4 → (bloque medio 256×4×4) → … → 3×32×32 (tabla real abajo) |
| **Parámetros entrenables** | Todos los de la U-Net (≈35.7 M) durante el entrenamiento original; **0** en esta práctica (congelado). El scheduler no tiene parámetros. |
| **Salida** | ε̂_θ(x_t, t): tensor 3×32×32 (estimación del ruido contenido en x_t) |
| **Flujo del aprendizaje** | forward q → x_t → U-Net → ε̂ → pérdida ‖ε−ε̂‖² → gradiente → actualización de pesos |
""")

    @st.cache_resource(show_spinner="Instanciando la U-Net de referencia (pesos aleatorios, solo para contar capas)…")
    def reference_unet_table():
        import torch
        from diffusers import UNet2DModel
        u = UNet2DModel(**theory.REFERENCE_UNET_CONFIG).eval()
        rows = core.layer_shapes(u)
        total = sum(p.numel() for p in u.parameters())
        return rows, total

    rows, total = None, None
    src = ""
    ls = load_json("layer_shapes.json")
    if ls and model_info:
        rows, total, src = ls, model_info["params_total"], "checkpoint real (results/layer_shapes.json)"
    elif core.torch_available():
        rows, total = reference_unet_table()
        src = "arquitectura de referencia instanciada localmente (pesos aleatorios; mismas formas que el checkpoint)"
    if rows:
        dfp = pd.DataFrame(rows)
        dfp["forma_salida (C×H×W)"] = dfp["forma_salida"].apply(lambda s: "×".join(map(str, s)))
        dfp = dfp[["bloque", "forma_salida (C×H×W)", "parametros"]].rename(columns={"parametros": "parámetros"})
        st.dataframe(dfp, hide_index=True)
        st.caption(f"Fuente: {src}. Parámetros totales U-Net: **{total:,}** ({total/1e6:.2f} M). "
                   "La suma por bloque no incluye el MLP del timestep (time_embedding).")
    else:
        st.info("Instale `torch` y `diffusers` para ver aquí la tabla de formas por bloque (se calcula con un forward real).")

    st.subheader("6. DDPM vs DDIM — tabla de compromisos")
    st.markdown("""
| Aspecto | DDPM | DDIM (η = 0) |
|---|---|---|
| Naturaleza | Estocástico (ruido nuevo en cada paso) | Determinista |
| Reproducibilidad con semilla | Sí, si también se fija el generador del ruido por paso | Sí, depende solo de x_T |
| Pasos necesarios | Muchos (el entrenamiento supone 1000); con pocos pasos suele degradarse | Tolera menos pasos |
| Diversidad entre muestras | Aporta variabilidad extra por el ruido de cada paso | Solo la de x_T |
| Costo por paso | Casi igual (la U-Net domina) | Casi igual |
| **Qué se mide aquí** | tiempo, memoria, diversidad, ruido residual y calificación visual — con las mismas semillas (pestaña *Evaluación*) | idem |
""")
    st.caption("Las celdas 'suele…' son expectativas de la literatura, no resultados de este trabajo: se contrastan con los datos propios en *Evaluación*.")


# ============================================================================= 3. DIFUSIÓN DIRECTA
with tabs[2]:
    st.header("Difusión directa interactiva")
    st.write("Esta pestaña **no necesita el modelo**: usa solo el calendario de ruido (β lineal, T = 1000).")
    betas = theory.linear_betas(); abar = theory.alpha_bars(betas)
    c1, c2 = st.columns([1, 2])
    with c1:
        src = st.radio("Imagen x₀", ["Imagen de demostración (sintética)", "Subir imagen"])
        x0 = theory.demo_image(32)
        if src == "Subir imagen":
            up = st.file_uploader("PNG/JPG (se reduce a 32×32 RGB)", type=["png", "jpg", "jpeg"])
            if up is not None:
                from PIL import Image
                im = Image.open(up).convert("RGB").resize((32, 32), Image.LANCZOS)
                x0 = np.asarray(im, np.float32) / 127.5 - 1
        t = st.slider("Timestep t", 0, 999, 300)
        seed = st.number_input("Semilla de ε", 0, 10_000, 0)
        eps = np.random.default_rng(int(seed)).standard_normal(x0.shape)
        xt = theory.q_sample(x0, t, eps, abar)
        st.metric("ᾱ_t (señal preservada)", f"{abar[t]:.4f}")
        st.metric("√ᾱ_t · x₀  /  √(1−ᾱ_t) · ε", f"{np.sqrt(abar[t]):.3f}  /  {np.sqrt(1-abar[t]):.3f}")
        st.metric("SNR = ᾱ_t/(1−ᾱ_t)", f"{theory.snr(abar)[t]:.4g}")
    with c2:
        a, b, c = st.columns(3)
        grid = lambda im: core.make_grid([theory.to_img01(im)], 1, scale=8, pad=0)
        a.image(grid(x0), caption="x₀ (original)", width=190)
        b.image(grid(np.clip(eps / 2, -1, 1)), caption="ε ∼ N(0, I)", width=190)
        c.image(grid(xt), caption=f"x_t (t={t})", width=190)
        st.markdown("**Secuencia de corrupción** (mismo ε):")
        tt = [0, 25, 50, 100, 200, 350, 500, 750, 999]
        st.image(core.make_grid([theory.to_img01(theory.q_sample(x0, k, eps, abar)) for k in tt], len(tt), scale=3),
                 caption="t = " + ", ".join(map(str, tt)))
    fig, axs = plt.subplots(1, 2, figsize=(11, 3))
    axs[0].plot(np.sqrt(abar), label=r"$\sqrt{\bar\alpha_t}$ (peso de la señal)")
    axs[0].plot(np.sqrt(1 - abar), label=r"$\sqrt{1-\bar\alpha_t}$ (peso del ruido)")
    axs[0].axvline(t, color="k", ls=":"); axs[0].legend(); axs[0].set_xlabel("t"); axs[0].grid(alpha=.2)
    axs[1].semilogy(theory.snr(abar), color="#dc2626"); axs[1].axvline(t, color="k", ls=":")
    axs[1].set_xlabel("t"); axs[1].set_title("Relación señal/ruido (escala log)"); axs[1].grid(alpha=.2)
    st.pyplot(fig); plt.close(fig)

    st.subheader("Una operación explicada con números (un solo píxel)")
    px, e1 = 0.6, -0.8
    tt1 = 500
    val = np.sqrt(abar[tt1]) * px + np.sqrt(1 - abar[tt1]) * e1
    st.markdown(f"Sea x₀ = {px}, ε = {e1}, t = {tt1}.  ᾱ₅₀₀ = **{abar[tt1]:.5f}** → √ᾱ = **{np.sqrt(abar[tt1]):.5f}**, √(1−ᾱ) = **{np.sqrt(1-abar[tt1]):.5f}**")
    st.latex(rf"x_{{500}}=\sqrt{{\bar\alpha_{{500}}}}\,x_0+\sqrt{{1-\bar\alpha_{{500}}}}\,\varepsilon={np.sqrt(abar[tt1]):.5f}\cdot{px}+{np.sqrt(1-abar[tt1]):.5f}\cdot({e1})={val:.5f}")
    if st.button("Verificar: aplicar 501 pasos de q(x_k | x_{k−1}) vs. fórmula cerrada"):
        mi, si, mc, sc = theory.iterative_vs_closed_form(px, tt1)
        st.success(f"Iterativo (20 000 muestras): media {mi:.3f}, desv. {si:.3f}  |  Forma cerrada: media {mc:.3f}, desv. {sc:.3f}. "
                   f"Teórico: media {np.sqrt(abar[tt1])*px:.3f}, desv. {np.sqrt(1-abar[tt1]):.3f}.")


# ============================================================================= 4. GALERÍA
with tabs[3]:
    st.header("Galería y demostración (resultados precalculados)")
    st.warning("El modelo es **incondicional**: no existe 'prompt'. Las imágenes no se generan desde una descripción textual; "
               "cada una es una muestra aleatoria determinada por la semilla, el scheduler y los pasos.")
    if not run_ready():
        st.info("Todavía no hay resultados. Genere la galería con `python run_experiments.py` (o en la pestaña *Ejecutar experimentos*) y recargue.")
    else:
        avail = sorted({(r.scheduler, int(r.steps)) for r in timings.itertuples()}, key=lambda c: (c[0], c[1]))
        seeds_av = sorted(timings.seed.unique().tolist())
        st.subheader("A. Grilla: mismas semillas, distinto número de pasos")
        nshow = st.slider("Semillas a mostrar", 2, len(seeds_av), min(8, len(seeds_av)))
        fig, axs = plt.subplots(len(avail), 1, figsize=(1.55 * nshow, 1.55 * len(avail) + 0.4))
        axs = np.atleast_1d(axs)
        for ax, (s, n) in zip(axs, avail):
            ax.imshow(core.make_grid([get_img(s, n, sd) for sd in seeds_av[:nshow]], nshow, scale=2, pad=1))
            ax.set_yticks([]); ax.set_xticks([]); ax.set_ylabel(core.cfg_label(s, n), rotation=0, ha="right", va="center", fontsize=9)
            for sp in ax.spines.values(): sp.set_visible(False)
        axs[-1].set_xlabel("semillas: " + ", ".join(map(str, seeds_av[:nshow])), fontsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        st.subheader("B. Selector de configuración, pasos y semilla")
        c1, c2, c3 = st.columns(3)
        sch = c1.selectbox("Scheduler", sorted({a for a, _ in avail}))
        stp = c2.selectbox("Pasos", sorted({b for a, b in avail if a == sch}))
        sd = c3.selectbox("Semilla", seeds_av)
        row = timings[(timings.scheduler == sch) & (timings.steps == stp) & (timings.seed == sd)]
        base_im = get_img(*core.BASELINE, sd)
        cc = st.columns([1, 1, 2])
        cc[0].image(core.make_grid([get_img(sch, stp, sd)], 1, scale=8, pad=0), caption=f"{core.cfg_label(sch, stp)} · semilla {sd}", width=256)
        if base_im is not None:
            cc[1].image(core.make_grid([base_im], 1, scale=8, pad=0), caption=f"Línea base {core.cfg_label(*core.BASELINE)} · misma semilla", width=256)
        if len(row):
            cc[2].markdown(f"**Tiempo de esta imagen:** {row.seconds.iloc[0]:.3f} s  \n**Dispositivo:** {row.device.iloc[0]}")
        if per_img is not None:
            pr = per_img[(per_img.scheduler == sch) & (per_img.steps == stp) & (per_img.seed == sd)]
            if len(pr):
                r = pr.iloc[0]
                cc[2].markdown(f"TV (ruido residual): {r.tv:.3f} · contraste: {r.contrast:.3f}  \n"
                               f"SSIM vs. línea base: {r.get('ssim_vs_baseline', float('nan')):.3f} · PSNR: {r.get('psnr_vs_baseline', float('nan')):.2f} dB")
        cc[2].caption("SSIM/PSNR miden parecido con *otra* imagen generada (línea base), no realismo.")

        st.subheader("C. Mosaicos de diversidad por configuración")
        cols = st.columns(4)
        for k, (s, n) in enumerate(avail):
            cols[k % 4].image(core.make_grid([get_img(s, n, x) for x in seeds_av], 4, scale=2, pad=1),
                              caption=f"{core.cfg_label(s, n)} · {len(seeds_av)} semillas")

        st.subheader("D. Pasos de denoising (x_t y estimación x̂₀)")
        trajs = sorted((RES / "trajectories").glob("*.npz")) if (RES / "trajectories").exists() else []
        if trajs:
            pick = st.selectbox("Trayectoria", trajs, format_func=lambda p: p.stem.replace("_", "-", 1).upper())
            z = np.load(pick)
            k = len(z["t"])
            fig, axs = plt.subplots(2, k + 1, figsize=(1.45 * (k + 1), 3.4))
            for j in range(k):
                axs[0, j].imshow(z["xt"][j]); axs[1, j].imshow(z["x0"][j])
                axs[0, j].set_title(f"t={int(z['t'][j])}", fontsize=8)
            axs[0, k].imshow(z["final"]); axs[1, k].imshow(z["final"]); axs[0, k].set_title("final", fontsize=8)
            axs[0, 0].set_ylabel("x_t", fontsize=9); axs[1, 0].set_ylabel("x̂₀", fontsize=9)
            for a in axs.ravel(): a.set_xticks([]); a.set_yticks([])
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            st.caption("Fila superior: la imagen ruidosa que entra a la U-Net. Fila inferior: lo que la red 'cree' que es la imagen limpia en ese instante (x̂₀). "
                       "Primero aparece la estructura gruesa (colores, formas) y al final los detalles.")
        else:
            st.info("No hay trayectorias guardadas todavía.")


# ============================================================================= 5. EVALUACIÓN
def auto_findings(S: pd.DataFrame, P: pd.DataFrame | None):
    out = []
    for sch in core.SCHEDULERS:
        d = S[S.scheduler == sch].sort_values("steps")
        if len(d) >= 3:
            a, b = np.polyfit(d.steps, d.sec_mean, 1)
            pred = a * d.steps + b
            r2 = 1 - ((d.sec_mean - pred) ** 2).sum() / max(((d.sec_mean - d.sec_mean.mean()) ** 2).sum(), 1e-12)
            out.append(f"**Latencia vs. pasos ({sch}):** el tiempo crece de forma {'casi lineal' if r2 > .98 else 'no estrictamente lineal'} "
                       f"con el número de pasos (≈ {1000*a:.1f} ms por paso adicional, R² = {r2:.3f}); cada paso cuesta una evaluación de la U-Net.")
    m = S.pivot(index="steps", columns="scheduler", values="sec_mean")
    if {"DDPM", "DDIM"} <= set(m.columns) and m.dropna().shape[0]:
        ratio = (m.DDIM / m.DDPM).dropna()
        out.append(f"**Costo del scheduler:** a igual número de pasos, DDIM tardó entre {ratio.min():.2f}× y {ratio.max():.2f}× el tiempo de DDPM: "
                   "la diferencia de costo entre schedulers es pequeña frente a la de cambiar el número de pasos.")
    b = S[(S.scheduler == core.BASELINE[0]) & (S.steps == core.BASELINE[1])]
    if len(b):
        bt = b.sec_mean.iloc[0]
        fast = S.sort_values("sec_mean").iloc[0]
        out.append(f"**Ahorro frente a la línea base:** {core.cfg_label(fast.scheduler, fast.steps)} es la más rápida ({fast.sec_mean:.3f} s/img), "
                   f"{bt / fast.sec_mean:.1f}× más rápida que {core.cfg_label(*core.BASELINE)} ({bt:.3f} s/img). Lo que falta por decidir es cuánta calidad se pierde (ver calificaciones).")
    if "ssim_vs_baseline_mean" in S:
        d = S[~((S.scheduler == core.BASELINE[0]) & (S.steps == core.BASELINE[1]))].dropna(subset=["ssim_vs_baseline_mean"])
        if len(d):
            hi, lo = d.sort_values("ssim_vs_baseline_mean").iloc[-1], d.sort_values("ssim_vs_baseline_mean").iloc[0]
            out.append(f"**Consistencia con la línea base (SSIM, misma semilla):** la más parecida es {core.cfg_label(hi.scheduler, hi.steps)} ({hi.ssim_vs_baseline_mean:.3f}) "
                       f"y la menos parecida {core.cfg_label(lo.scheduler, lo.steps)} ({lo.ssim_vs_baseline_mean:.3f}). Es consistencia entre configuraciones, no calidad.")
    if "div_pairwise_rms" in S:
        hi, lo = S.sort_values("div_pairwise_rms").iloc[-1], S.sort_values("div_pairwise_rms").iloc[0]
        out.append(f"**Diversidad (distancia RMS por pares):** mayor en {core.cfg_label(hi.scheduler, hi.steps)} ({hi.div_pairwise_rms:.3f}), menor en {core.cfg_label(lo.scheduler, lo.steps)} ({lo.div_pairwise_rms:.3f}).")
    if "tv_mean" in S:
        hi = S.sort_values("tv_mean").iloc[-1]
        out.append(f"**Ruido residual (variación total):** la mayor se observa en {core.cfg_label(hi.scheduler, hi.steps)} ({hi.tv_mean:.3f}); valores altos suelen indicar granulado de alta frecuencia sin eliminar.")
    return out


def rating_table(split="validation"):
    p = RES / "ratings.csv"
    if not p.exists():
        return None
    r = pd.read_csv(p)
    if "split" in r:
        r = r[r.split == split]
    return r if len(r) else None


with tabs[4]:
    st.header("Evaluación y comparación")
    if summary is None:
        st.info("Sin resultados: ejecute los experimentos primero.")
    else:
        S = summary.copy()
        S["label"] = [core.cfg_label(a, b) for a, b in zip(S.scheduler, S.steps)]
        st.subheader("Tabla de latencia, memoria, diversidad y similitud")
        base_t = S[(S.scheduler == core.BASELINE[0]) & (S.steps == core.BASELINE[1])].sec_mean
        S["costo_rel_vs_base"] = S.sec_mean / base_t.iloc[0] if len(base_t) else np.nan
        show_cols = {"label": "Config", "n_images": "n", "sec_mean": "s/img (media)", "sec_std": "desv.", "sec_ci95": "IC95% ±",
                     "sec_per_step_ms": "ms/paso", "costo_rel_vs_base": "tiempo rel. a DDPM-100", "peak_cuda_mb": "pico CUDA (MB)",
                     "rss_mb_max": "RSS máx (MB)", "div_pairwise_rms": "diversidad RMS", "div_pixel_std": "desv. píxel entre semillas",
                     "tv_mean": "TV (ruido res.)", "contrast_mean": "contraste", "ssim_vs_baseline_mean": "SSIM vs base",
                     "psnr_vs_baseline_mean": "PSNR vs base (dB)"}
        for c_ in ("peak_cuda_mb", "rss_mb_max"):
            S[c_] = pd.to_numeric(S[c_], errors="coerce")
        T = S[[c for c in show_cols if c in S and not S[c].isna().all()]].rename(columns=show_cols)
        st.dataframe(T.style.format({c: "{:.3f}" for c in T.columns if c not in ("Config", "n")}, na_rep="—"), hide_index=True)
        st.caption("Memoria en CPU = RSS del proceso (aproximada; su pico es monótono). En GPU = máximo asignado por PyTorch (MB). "
                   "Peso del modelo: " + (f"{model_info['weights_mb']} MB" if model_info else "—") + ".")

        st.subheader("Gráficos")
        c1, c2 = st.columns(2)
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        for sch, col in zip(core.SCHEDULERS, ("#2563eb", "#dc2626")):
            d = S[S.scheduler == sch].sort_values("steps")
            ax.errorbar(d.steps, d.sec_mean, yerr=d.sec_ci95, marker="o", capsize=3, label=sch, color=col)
        ax.set_xlabel("Pasos de inferencia"); ax.set_ylabel("Segundos por imagen (media ± IC95%)"); ax.grid(alpha=.25); ax.legend()
        ax.set_title("Tiempo promedio por imagen vs pasos")
        c1.pyplot(fig); plt.close(fig)
        if "div_pairwise_rms" in S:
            fig, ax = plt.subplots(figsize=(5.6, 3.6))
            for sch, col in zip(core.SCHEDULERS, ("#2563eb", "#dc2626")):
                d = S[S.scheduler == sch].sort_values("steps")
                ax.plot(d.steps, d.div_pairwise_rms, marker="o", color=col, label=f"{sch} · diversidad RMS")
                if "tv_mean" in d:
                    ax.plot(d.steps, d.tv_mean, marker="s", ls="--", color=col, alpha=.6, label=f"{sch} · TV")
            ax.set_xlabel("Pasos de inferencia"); ax.grid(alpha=.25); ax.legend(fontsize=7); ax.set_title("Diversidad y ruido residual vs pasos")
            c2.pyplot(fig); plt.close(fig)

        if per_img is not None and "ssim_vs_baseline" in per_img:
            fig, ax = plt.subplots(figsize=(8, 3))
            labs = sorted({(a, int(b)) for a, b in zip(per_img.scheduler, per_img.steps)}, key=lambda c: (c[0], c[1]))
            ax.boxplot([per_img[(per_img.scheduler == a) & (per_img.steps == b)].ssim_vs_baseline.dropna() for a, b in labs],
                       tick_labels=[core.cfg_label(a, b) for a, b in labs])
            ax.set_ylabel("SSIM vs DDPM-100 (misma semilla)"); ax.grid(alpha=.25); ax.tick_params(axis="x", labelsize=8)
            st.pyplot(fig); plt.close(fig)

        st.subheader("Hallazgos calculados automáticamente (a partir de sus datos reales)")
        for k, f in enumerate(auto_findings(S, per_img), 1):
            st.markdown(f"{k}. {f}")
        st.caption("Estos textos solo reportan lo medido. La **interpretación** (por qué ocurre, qué significa) debe redactarla el grupo:")
        interp_p = RES / "interpretacion.md"
        txt = st.text_area("Interpretación del grupo (≥ 3 hallazgos interpretados)", interp_p.read_text(encoding="utf-8") if interp_p.exists() else "", height=160)
        if st.button("Guardar interpretación"):
            try:
                interp_p.write_text(txt, encoding="utf-8"); st.success("Guardado en results/interpretacion.md")
            except OSError as e:
                st.warning(f"No se pudo escribir en disco ({e}). Use el botón de descarga.")
        st.download_button("Descargar interpretación (.md)", txt, file_name="interpretacion.md")
        if IS_CLOUD:
            st.caption("Aviso: en Streamlit Cloud el disco es temporal; lo que guarde aquí se pierde al reiniciar. Descargue el archivo y súbalo a GitHub (carpeta results/).")

        st.subheader("Calidad visual (calificación a ciegas) y selección de configuración")
        R = rating_table("validation")
        if R is None:
            st.info("Aún no hay calificaciones. Use la pestaña **Calificación a ciegas** (escala 1–5). Sin ellas la app no selecciona configuración.")
        else:
            rt = R.groupby(["scheduler", "steps"]).score.agg(["mean", "std", "count"]).reset_index()
            rt["Config"] = [core.cfg_label(a, b) for a, b in zip(rt.scheduler, rt.steps)]
            st.dataframe(rt[["Config", "mean", "std", "count"]].rename(columns={"mean": "media", "std": "desv.", "count": "n calificaciones"}).style.format({"media": "{:.2f}", "desv.": "{:.2f}"}), hide_index=True)
            tol = st.slider("Tolerancia de calidad (puntos respecto a la mejor)", 0.0, 1.5, 0.5, 0.1,
                            help="Regla pre-declarada: menor tiempo entre las configuraciones con media ≥ mejor − tolerancia")
            sel = core.select_config(summary, R, tol)
            if sel:
                st.success(f"**Configuración seleccionada: {core.cfg_label(sel['scheduler'], sel['steps'])}** · calificación media {sel['rating_mean']:.2f} · "
                           f"{sel['sec_mean']:.3f} s/img. Candidatas dentro de la tolerancia: "
                           + ", ".join(core.cfg_label(a, b) for a, b in sel["candidates"]))

        st.subheader("Tabla de compromisos")
        st.markdown("""
| Si priorizo… | Elijo… | Costo / riesgo |
|---|---|---|
| Mínimo tiempo | Menos pasos | Posible pérdida de calidad / más ruido residual (verificar en los datos de arriba) |
| Máxima calidad observada | La config. mejor calificada | Más segundos por imagen |
| Reproducibilidad estricta | DDIM (η=0) | Menor variabilidad de muestreo (la diversidad viene solo de x_T) |
| Diversidad estocástica | DDPM | Requiere fijar también el generador del ruido de cada paso para reproducir |
""")

        st.subheader("Evaluación final en semillas de prueba (nunca usadas para elegir)")
        ft = load_csv("timings.csv", FINAL)
        fs = load_csv("summary.csv", FINAL)
        if fs is None:
            st.info("Pendiente: tras seleccionar la configuración, ejecute la **prueba final** en la pestaña *Ejecutar experimentos*.")
        else:
            fs = fs.copy(); fs["Config"] = [core.cfg_label(a, b) for a, b in zip(fs.scheduler, fs.steps)]
            st.dataframe(fs[["Config", "n_images", "sec_mean", "sec_ci95", "div_pairwise_rms", "tv_mean"]].style.format(precision=3), hide_index=True)
            fr = rating_table("test")
            if fr is not None:
                g = fr.groupby(["scheduler", "steps"]).score.agg(["mean", "count"]).reset_index()
                st.dataframe(g, hide_index=True)
            cols = st.columns(len(fs))
            for col, r in zip(cols, fs.itertuples()):
                col.image(core.make_grid([get_img(r.scheduler, int(r.steps), x, FINAL) for x in core.TEST_SEEDS], 5, scale=2, pad=1),
                          caption=f"{r.Config} · semillas de prueba 200–209")
        st.caption("FID/KID: no se calculan; con 10–16 muestras serían estadísticamente inválidos (se necesitarían ≥ 10 000 y una red de características). Ver *Limitaciones*.")


# ============================================================================= 6. CALIFICACIÓN
with tabs[5]:
    st.header("Calificación visual a ciegas")
    st.write("Escala de calidad definida por evaluadores. Se muestra **una imagen sin indicar su configuración**, en orden aleatorio, para evitar sesgo de confirmación.")
    st.table(pd.DataFrame({"Puntaje": list(content.RATING_SCALE), "Criterio": list(content.RATING_SCALE.values())}))
    if not run_ready():
        st.info("Primero genere imágenes.")
    else:
        split = st.radio("Conjunto a calificar", ["validation", "test"], horizontal=True,
                         format_func=lambda s: "Validación (semillas 100–115)" if s == "validation" else "Prueba final (semillas 200–209)")
        base = RES if split == "validation" else FINAL
        tt = load_csv("timings.csv", base)
        who = st.text_input("Nombre del evaluador", key="evaluator")
        if tt is None:
            st.info("No hay imágenes en ese conjunto todavía.")
        elif who.strip():
            items = [(r.scheduler, int(r.steps), int(r.seed)) for r in tt.itertuples()]
            rp = RES / "ratings.csv"
            prev = pd.read_csv(rp) if rp.exists() else pd.DataFrame(columns=["timestamp", "evaluator", "split", "scheduler", "steps", "seed", "score"])
            if "split" not in prev: prev["split"] = "validation"
            done = {(r.scheduler, int(r.steps), int(r.seed)) for r in prev[(prev.evaluator == who.strip()) & (prev.split == split)].itertuples()}
            pending = [i for i in items if i not in done]
            st.progress(1 - len(pending) / max(len(items), 1), text=f"{len(items) - len(pending)} / {len(items)} calificadas por {who.strip()}")
            if not pending:
                st.success("Ha calificado todas las imágenes de este conjunto.")
            else:
                key = (who.strip(), split)
                if st.session_state.get("rate_key") != key or st.session_state.get("rate_item") not in pending:
                    st.session_state["rate_key"] = key
                    st.session_state["rate_item"] = random.choice(pending)
                item = st.session_state["rate_item"]
                cA, cB = st.columns([1, 1])
                cA.image(core.make_grid([get_img(*item, base)], 1, scale=10, pad=0), width=320)
                with cB.form("rate_form", clear_on_submit=True):
                    sc = st.radio("Calidad percibida", list(content.RATING_SCALE), index=None,
                                  format_func=lambda k: f"{k} · {content.RATING_SCALE[k]}")
                    ok = st.form_submit_button("Guardar y siguiente")
                if ok:
                    if sc is None:
                        st.warning("Elija un puntaje.")
                    else:
                        new = pd.DataFrame([{"timestamp": datetime.now().isoformat(timespec="seconds"), "evaluator": who.strip(),
                                             "split": split, "scheduler": item[0], "steps": item[1], "seed": item[2], "score": sc}])
                        try:
                            pd.concat([prev, new]).to_csv(rp, index=False)
                        except OSError as e:
                            st.error(f"No se pudo guardar: {e}"); st.stop()
                        st.session_state.pop("rate_item", None)
                        st.rerun()
        else:
            st.info("Escriba su nombre para comenzar.")
        if (RES / "ratings.csv").exists():
            a = pd.read_csv(RES / "ratings.csv")
            st.caption(f"Calificaciones totales guardadas: {len(a)} · evaluadores: {a.evaluator.nunique()}")
            st.download_button("Descargar ratings.csv", a.to_csv(index=False), file_name="ratings.csv", mime="text/csv")
        if IS_CLOUD:
            st.warning("En Streamlit Cloud las calificaciones se guardan en un disco temporal y se pierden al reiniciar la app. "
                       "Descargue ratings.csv periódicamente y súbalo a GitHub (results/ratings.csv) para conservarlas.")


# ============================================================================= 7. EN VIVO
with tabs[6]:
    st.header("Generación en vivo (inferencia)")
    st.info("Modelo incondicional: **no hay texto de entrada**. Controle semilla, pasos y scheduler. En CPU, 100 pasos pueden tardar varios segundos por imagen.")
    if not core.torch_available():
        st.error("Faltan `torch` y/o `diffusers`. Instale `requirements.txt` o use la galería precalculada.")
        st.caption("Si ya los instaló y falla al importar, suele ser un conflicto numpy/transformers: pruebe un entorno limpio (ver README, sección Solución de problemas).")
    else:
        import torch
        mid, offl = model_source_ui("live")
        dev = st.selectbox("Dispositivo", ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"])

        @st.cache_resource(show_spinner="Cargando modelo…")
        def get_pipe(m, d, o):
            return core.load_pipeline(m, d, o)

        if st.button("Cargar modelo", disabled=mid is None):
            try:
                st.session_state["pipe"] = get_pipe(mid, dev, offl)
                st.session_state["pipe_name"] = (mid, dev)
            except Exception as e:
                st.error(f"No se pudo cargar el modelo: {e}")
                st.caption("Si el mensaje menciona numpy/transformers: `pip install -U transformers peft accelerate` o use un entorno virtual limpio.")
                st.caption("Sin red a huggingface.co: descargue el repositorio manualmente (`huggingface-cli download google/ddpm-cifar10-32 --local-dir ddpm-cifar10-32`) y escriba esa carpeta arriba.")
        pipe = st.session_state.get("pipe")
        if pipe is not None:
            ms = core.model_summary(pipe)
            st.success(f"Cargado: `{st.session_state['pipe_name'][0]}` en {st.session_state['pipe_name'][1]} · {ms['unet_class']} · "
                       f"{ms['params_total']/1e6:.2f} M parámetros · pesos {ms['weights_mb']} MB · scheduler del checkpoint: {ms['scheduler_class_in_checkpoint']}")
            g1, g2, g3, g4 = st.columns(4)
            s_name = g1.selectbox("Scheduler", core.SCHEDULERS, key="live_s")
            s_steps = g2.number_input("Pasos", 1, 1000, 50, key="live_n")
            s_seed = g3.number_input("Semilla inicial", 0, 10**6, 100, key="live_seed")
            s_num = g4.number_input("N° de imágenes", 1, 16, 4, key="live_k")
            want_traj = st.checkbox("Mostrar proceso de denoising de la primera imagen", True)
            if st.button("Generar"):
                ims, secs = [], []
                bar = st.progress(0.0)
                for k in range(int(s_num)):
                    r = core.sample_one(pipe.unet, pipe.scheduler, s_name, int(s_steps), int(s_seed) + k, pipe._ddpm_device,
                                        n_snapshots=8 if (want_traj and k == 0) else 0)
                    if k == 0 and want_traj:
                        traj = r["snapshots"]
                        r2 = core.sample_one(pipe.unet, pipe.scheduler, s_name, int(s_steps), int(s_seed), pipe._ddpm_device)
                        secs.append(r2["seconds"])   # el tiempo se mide sin capturas
                    else:
                        secs.append(r["seconds"])
                    ims.append(r["image"]); bar.progress((k + 1) / int(s_num))
                st.image(core.make_grid(ims, min(8, len(ims)), scale=5), caption=f"{s_name} · {int(s_steps)} pasos · semillas {int(s_seed)}…{int(s_seed)+len(ims)-1}")
                st.write(f"**Tiempo medio:** {np.mean(secs):.3f} s/imagen (n={len(secs)}) · dispositivo {pipe._ddpm_device}")
                if want_traj and traj:
                    fig, axs = plt.subplots(2, len(traj), figsize=(1.5 * len(traj), 3.3))
                    for j, sn in enumerate(traj):
                        axs[0, j].imshow(sn["xt"]); axs[1, j].imshow(sn["x0"]); axs[0, j].set_title(f"t={sn['t']}", fontsize=8)
                    axs[0, 0].set_ylabel("x_t"); axs[1, 0].set_ylabel("x̂₀")
                    for a in axs.ravel(): a.set_xticks([]); a.set_yticks([])
                    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            st.subheader("Comprobación de reproducibilidad")
            if st.button("Ejecutar prueba (misma semilla dos veces)"):
                rep = core.reproducibility_check(pipe.unet, pipe.scheduler, pipe._ddpm_device, int(s_seed), 25)
                st.json(rep)
                st.caption("En CPU se espera igualdad exacta. En GPU algunos kernels no son bit a bit deterministas: se reporta la diferencia máxima real.")


# ============================================================================= 8. EJECUTAR
with tabs[7]:
    st.header("Ejecutar el protocolo experimental")
    st.markdown("""Genera **8 configuraciones × 16 semillas**, mide tiempo/memoria, guarda imágenes, trayectorias, métricas y la comprobación de reproducibilidad.
Se recomienda ejecutar por terminal (o en Google Colab con GPU) y luego abrir esta app:
```bash
pip install -r requirements.txt
python run_experiments.py              # protocolo completo (reanudable)
streamlit run app.py
```
""")
    if not core.torch_available():
        st.error("Instale `torch` y `diffusers` para ejecutar desde aquí.")
    elif IS_CLOUD and not st.checkbox("Entiendo que en la nube (CPU compartida y límite de memoria) esto puede tardar horas o cortarse; prefiero ejecutarlo en mi PC"):
        st.info("Los resultados ya vienen precalculados en `results/`. Para repetir el protocolo, hágalo en local (`python run_experiments.py`) y suba la carpeta `results/` a GitHub.")
    else:
        import torch
        mid, off = model_source_ui("run")
        c2, c3 = st.columns(2)
        dev = c2.selectbox("Dispositivo", ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"], key="run_dev")
        nse = c3.number_input("Semillas (10–16)", 10, 16, 16)
        if st.button("Ejecutar validación (8 configs)", disabled=mid is None):
            bar = st.progress(0.0, text="Cargando modelo…"); logbox = st.empty()
            lines = []
            def lg(*a):
                lines.append(" ".join(map(str, a))); logbox.code("\n".join(lines[-12:]))
            try:
                core.run_experiments(RES, mid, seeds=core.COMPARISON_SEEDS[:int(nse)], device=dev, local_files_only=off,
                                     progress=lambda f, t: bar.progress(min(f, 1.0), text=t), log=lg)
                st.success("Listo. Recargue la página para ver resultados."); st.cache_data.clear()
            except Exception as e:
                st.error(f"Falló la ejecución: {e}")
        st.markdown("---")
        st.markdown("**Prueba final** (tras elegir configuración): evalúa línea base y configuración elegida en semillas 200–209.")
        csel = st.selectbox("Configuración elegida", core.CONFIGS, format_func=lambda c: core.cfg_label(*c), key="final_sel")
        if st.button("Ejecutar prueba final", disabled=mid is None):
            bar = st.progress(0.0, text="…")
            try:
                core.run_final_test(RES, csel, mid, dev, off, progress=lambda f, t: bar.progress(min(f, 1.0), text=t), log=lambda *a: None)
                st.success("Prueba final lista."); st.cache_data.clear()
            except Exception as e:
                st.error(f"Falló: {e}")


# ============================================================================= 9. 16 ETAPAS
with tabs[8]:
    st.header("Metodología obligatoria: 16 etapas")
    st.caption("Cada etapa tiene subtítulo, explicación y una evidencia verificable. 'Pendiente' = requiere ejecutar los experimentos.")

    def ev_stage(n):
        has = run_ready()
        if n == 3:
            mc = RES / "model_card.md"
            if model_info:
                st.json({k: model_info.get(k) for k in ("model_id", "revision", "license", "card_saved")})
            if mc.exists(): st.caption("Model card guardada en results/model_card.md"); st.text(mc.read_text(encoding="utf-8")[:900])
            return model_info is not None
        if n == 4:
            st.table(pd.DataFrame({"Elemento": ["Ruido inicial", "Timestep", "Scheduler", "Salida"],
                                   "Detalle": ["x_T ∼ N(0,I), 3×32×32, desde torch.Generator('cpu').manual_seed(semilla)", "t ∈ {0..999}; subconjunto según pasos",
                                               "DDPM o DDIM (η=0)", "Imagen RGB 32×32 uint8"]}))
            return True
        if n == 5 and has:
            ims = [get_img(*core.BASELINE, s) for s in sorted(timings.seed.unique())[:12]]
            st.image(core.make_grid(ims, 6, scale=3), caption="Muestras de la línea base DDPM-100")
            arr = np.stack([i for i in ims if i is not None]).astype(float) / 255
            fig, ax = plt.subplots(figsize=(5, 2.4))
            for c, col in enumerate(("r", "g", "b")): ax.hist(arr[..., c].ravel(), bins=32, alpha=.5, color=col)
            ax.set_title("Histograma de píxeles por canal (muestras generadas)", fontsize=9); st.pyplot(fig); plt.close(fig)
            return True
        if n == 6:
            st.write({"exploración": core.EXPLORATION_SEEDS, "validación": core.COMPARISON_SEEDS, "prueba final": core.TEST_SEEDS})
            return True
        if n == 7 and env:
            st.json(env); return True
        if n == 8 and has:
            r = timings[(timings.scheduler == "DDPM") & (timings.steps == 100)]
            if len(r):
                st.image(core.make_grid([get_img("DDPM", 100, int(r.seed.iloc[0]))], 1, scale=6, pad=0), width=190,
                         caption=f"DDPM-100, semilla {int(r.seed.iloc[0])}: {r.seconds.iloc[0]:.3f} s")
                return True
        if n == 9:
            st.write("Ver diagrama y tabla de formas en la pestaña *Teoría y arquitectura*.")
            return True
        if n == 10:
            st.table(pd.DataFrame({"Parámetro": ["scheduler", "pasos", "semilla", "n imágenes", "resolución", "batch", "DDIM η", "espaciado"],
                                   "Valor": ["DDPM | DDIM", "25, 50, 100, 250", "100–115 (validación), 200–209 (prueba)", "16 / 10", "32×32", "1", "0.0", "leading"]}))
            return True
        if n == 11:
            rows = []
            try:
                from diffusers import DDIMScheduler, DDPMScheduler
                for sname, cls in (("DDPM", DDPMScheduler), ("DDIM", DDIMScheduler)):
                    for k in core.STEPS_GRID:
                        s_ = cls(num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02, beta_schedule="linear")
                        s_.set_timesteps(k)
                        rows.append({"scheduler": sname, "pasos": k, "soportado": "sí", "primer/último timestep": f"{int(s_.timesteps[0])} / {int(s_.timesteps[-1])}"})
            except Exception as e:
                rows = [{"scheduler": "—", "pasos": "—", "soportado": f"no verificable ({e})", "primer/último timestep": "—"}]
            st.table(pd.DataFrame(rows)); st.caption("Compatibilidad verificada instanciando cada scheduler con la configuración de entrenamiento.")
            return True
        if n == 12 and has:
            st.dataframe(timings.head(8)); st.caption(f"{len(timings)} filas en results/timings.csv"); return True
        if n == 13 and has:
            st.write(timings.groupby(["scheduler", "steps"]).seed.nunique().rename("semillas").reset_index())
            rep = load_json("reproducibility.json")
            if rep: st.json(rep)
            return True
        if n == 14:
            R = rating_table("validation")
            sel = core.select_config(summary, R) if (summary is not None and R is not None) else None
            if sel: st.success(f"Seleccionada: {core.cfg_label(sel['scheduler'], sel['steps'])}")
            return sel is not None
        if n == 15 and summary is not None:
            st.dataframe(summary[["scheduler", "steps", "sec_mean", "sec_ci95", "rss_mb_max", "peak_cuda_mb", "div_pairwise_rms"]].style.format(precision=3), hide_index=True)
            return True
        if n == 16:
            st.write("Pestañas *Galería y demo* y *Generación en vivo*. Recordatorio: modelo incondicional, **sin prompt de texto**.")
            return True
        return False

    for n, title, pide, resuelve in content.STAGES:
        with st.expander(f"{n:02d}. {title}"):
            st.markdown(f"**Qué pide la guía.** {pide}")
            st.markdown(f"**Cómo se resuelve.** {resuelve}")
            st.markdown("**Evidencia verificable.** " + "", unsafe_allow_html=True)
            ok = ev_stage(n)
            st.markdown(badge(bool(ok)), unsafe_allow_html=True)


# ============================================================================= 10. LIMITACIONES
with tabs[9]:
    st.header("Limitaciones, sesgos y condiciones donde no generaliza")
    for l in content.LIMITATIONS:
        st.markdown(f"- {l}")
    st.header("Preguntas de sustentación")
    for q, a in content.QA:
        with st.expander(q):
            st.markdown(a)
    st.header("Licencia, créditos y reproducción")
    st.markdown("""
- **Modelo:** `google/ddpm-cifar10-32` (Hugging Face). Revise la licencia en su *model card* (queda guardada en `results/model_card.md` al ejecutar). Se usa solo para investigación/docencia.
- **Código:** `diffusers` (Apache-2.0), `torch`, `numpy`, `pandas`, `scikit-image`, `matplotlib`, `streamlit`. Versiones exactas en `results/env.json` y `requirements.lock.txt`.
- **Imagen de demostración** de la pestaña *Difusión directa*: generada proceduralmente por el código (original).
- **Pasos de reproducción:** `pip install -r requirements.txt` → `python run_experiments.py` → `streamlit run app.py`. Semillas explícitas en `ddpm_core.py`.
""")
    st.header("Referencias (APA 7)")
    for r_ in content.REFERENCES:
        st.markdown(f"- {r_}")
