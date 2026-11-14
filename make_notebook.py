"""Genera notebook_didactico.ipynb (16 etapas). Uso: python make_notebook.py"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda t: C.append(nbf.v4.new_markdown_cell(t))
code = lambda t: C.append(nbf.v4.new_code_cell(t))

md("""# Evaluación de generación de imágenes con un DDPM preentrenado
**Grupo 10 · Sistemas Inteligentes y Machine Learning · Modelo `google/ddpm-cifar10-32`**

Notebook didáctico con las **16 etapas** de la guía. Cada etapa tiene un subtítulo, una explicación y una evidencia ejecutable.
Todo el cómputo reutiliza `ddpm_core.py` (el mismo que usa la app Streamlit). *Ninguna cifra está escrita a mano: se calcula al ejecutar.*

Variables de entorno opcionales: `DDPM_MODEL` (ID de Hugging Face o carpeta local), `DDPM_RESULTS` (carpeta de resultados), `DDPM_NSEEDS` (n.º de semillas, 10–16).""")
code("""import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
import ddpm_core as core, theory
MODEL   = os.environ.get("DDPM_MODEL", core.MODEL_ID)
RESULTS = os.environ.get("DDPM_RESULTS", "results")
NSEEDS  = int(os.environ.get("DDPM_NSEEDS", len(core.COMPARISON_SEEDS)))
LOCAL   = os.path.isdir(MODEL)
plt.rcParams["figure.dpi"] = 110""")

md("## 01. Definición del problema\nGenerar imágenes RGB **32×32** a partir de ruido con un modelo de difusión **preentrenado y congelado**, y estudiar cómo el número de pasos y el scheduler cambian **tiempo, memoria y calidad observable**. Restricciones: resolución fija y cómputo modesto (CPU o GPU pequeña). No se mide exactitud de clasificación.")
md("## 02. Tipo de aprendizaje\nGeneración **incondicional** por difusión (aprendizaje no supervisado generativo). En el entrenamiento original la red hace *regresión del ruido*; aquí solo hay **inferencia**. No hay etiquetas ni texto.")
md("## 03. Obtención y trazabilidad de datos\nSe descarga el checkpoint `google/ddpm-cifar10-32` (Hugging Face). Se registran revisión, parámetros y se guarda la model card (revise la licencia allí).")
code("""device = core.default_device()
pipe = core.load_pipeline(MODEL, device, local_files_only=LOCAL)
info = core.model_summary(pipe); info["revision"] = core._revision_of(MODEL)
print(json.dumps({k: info[k] for k in ("unet_class","scheduler_class_in_checkpoint","params_total","weights_mb","revision")}, indent=2))
print("Config scheduler del checkpoint:", info["scheduler_config"])""")
md("## 04. Entradas, salidas y objetivo\n- **Ruido inicial** `x_T ~ N(0,I)` de forma `(1,3,32,32)` fijado por la semilla.\n- **Timesteps** `t ∈ {0..999}`; el scheduler elige un subconjunto según los pasos.\n- **Red** U-Net → `ε̂(x_t,t)` `(1,3,32,32)`.\n- **Salida:** imagen RGB uint8 32×32.")
code("""x = pipe.unet.config
print("in_channels:", x.in_channels, "| sample_size:", x.sample_size, "| out_channels:", x.out_channels)
for s in core.SCHEDULERS:
    print(s, "25 pasos -> primeros timesteps:", core.timesteps_for(pipe.scheduler, s, 25)[:6])""")
md("## 05. Análisis exploratorio\nObservamos muestras **del propio modelo** (línea base, semillas de exploración). Es una descripción de lo generado, **no** de CIFAR-10 real.")
code("""r = [core.sample_one(pipe.unet, pipe.scheduler, *core.BASELINE[:1], core.BASELINE[1], s, device) for s in core.EXPLORATION_SEEDS]
plt.imshow(core.make_grid([a["image"] for a in r], 4, scale=4)); plt.axis("off"); plt.title("DDPM-100, semillas de exploración 0-3"); plt.show()
arr = np.stack([a["image"] for a in r]) / 255
print("media por canal:", arr.mean((0,1,2)).round(3), "| desv.:", arr.std((0,1,2)).round(3))""")
md("## 06. Separación experimental\nNo hay train local. Se **congelan** el modelo y los tres conjuntos de semillas **antes de ver resultados**:\n- *exploración* {0–3}: pruebas de humo (arriba);\n- *validación* {100–115}: comparar y seleccionar;\n- *prueba final* {200–209}: evalúa una sola vez la configuración elegida.")
code("""print("exploración:", core.EXPLORATION_SEEDS); print("validación :", core.COMPARISON_SEEDS); print("prueba     :", core.TEST_SEEDS)
print("modelo en eval():", not pipe.unet.training, "| parámetros con gradiente:", sum(p.requires_grad for p in pipe.unet.parameters()))""")
md("## 07. Preprocesamiento / entorno\nNo se preprocesan datos; se documenta el entorno (versiones, dispositivo, memoria).")
code("""env = core.env_info(device); print(json.dumps(env, indent=2, ensure_ascii=False))""")
md("## 08. Modelo de referencia\nMuestra base: **DDPM con 100 pasos**, que será la línea base.")
code("""base = core.sample_one(pipe.unet, pipe.scheduler, "DDPM", 100, core.COMPARISON_SEEDS[0], device)
plt.imshow(base["image"]); plt.axis("off"); plt.title(f"DDPM-100 · semilla {core.COMPARISON_SEEDS[0]} · {base['seconds']:.2f} s"); plt.show()""")
md("""## 09. Diseño de arquitectura
La U-Net estima el ruido ε̂. Forma por bloque (forward real) y papel del scheduler:
$$\\hat x_0=\\frac{x_t-\\sqrt{1-\\bar\\alpha_t}\\,\\varepsilon_\\theta}{\\sqrt{\\bar\\alpha_t}},\\qquad x_{t'}=f_{\\text{scheduler}}(x_t,\\hat\\varepsilon,t,t')$$""")
code("""pd.DataFrame(core.layer_shapes(pipe.unet, device))""")
code("""fig = theory.unet_diagram(info["unet_config"]); plt.show()""")
md("## 10. Configuración de inferencia\nControles: scheduler, pasos, semilla, n.º de imágenes. Resolución 32×32, batch 1, DDIM con η = 0, espaciado `leading`.")
code("""print({"scheduler": core.SCHEDULERS, "pasos": core.STEPS_GRID, "batch": 1, "ddim_eta": 0.0, "resolucion": core.RESOLUTION})""")
md("## 11. Plan de hiperparámetros\nPasos {25, 50, 100, 250} × {DDPM, DDIM} = 8 configuraciones (experimentos A–D). Se verifica compatibilidad; si un valor no se soporta se documenta.")
code("""for s, n in core.CONFIGS:
    try:
        ts = core.timesteps_for(pipe.scheduler, s, n); print(f"{s}-{n}: OK  {len(ts)} timesteps, {ts[0]} -> {ts[-1]}")
    except Exception as e:
        print(f"{s}-{n}: NO soportado -> {e}")""")
md("## 12. Ejecución y seguimiento\nSe ejecutan todas las configuraciones con **las mismas semillas**, guardando imagen y tiempo por muestra (con warm-up descartado). Es reanudable. *En CPU puede tardar varios minutos.*")
code("""seeds = core.COMPARISON_SEEDS[:NSEEDS]
core.run_experiments(RESULTS, MODEL, seeds=seeds, device=device, local_files_only=LOCAL, log=lambda *a: None)
t = pd.read_csv(f"{RESULTS}/timings.csv"); t.head()""")
md("## 13. Validación\nProtocolo repetible con 10–16 semillas por configuración y comprobación explícita de **reproducibilidad**.")
code("""print(json.dumps(json.load(open(f"{RESULTS}/reproducibility.json")), indent=2))
print(t.groupby(["scheduler","steps"]).seed.nunique().rename("semillas"))""")
md("## 14. Selección de configuración\nRegla **pre-declarada**: entre las configuraciones con calificación media ≥ mejor − 0.5, la de menor tiempo. Las calificaciones se recogen **a ciegas** en la app (pestaña *Calificación a ciegas*) y se guardan en `results/ratings.csv`.")
code("""S = pd.read_csv(f"{RESULTS}/summary.csv")
rp = f"{RESULTS}/ratings.csv"
R = pd.read_csv(rp) if os.path.exists(rp) else None
R = R[R.split == "validation"] if R is not None and "split" in R else R
print(core.select_config(S, R) or "Sin calificaciones humanas todavía: no se selecciona (no se inventa calidad).")""")
md("## 15. Evaluación final\nTabla de latencia, memoria y diversidad; la evaluación final de la configuración elegida frente a la línea base se hace en las semillas de prueba 200–209 (`core.run_final_test`). FID/KID **no** se calculan (muestra insuficiente).")
code("""S.round(4)""")
code("""fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for s, c in zip(core.SCHEDULERS, ("tab:blue", "tab:red")):
    d = S[S.scheduler == s].sort_values("steps")
    ax[0].errorbar(d.steps, d.sec_mean, yerr=d.sec_ci95, marker="o", capsize=3, label=s, color=c)
    ax[1].plot(d.steps, d.div_pairwise_rms, marker="o", label=s, color=c)
ax[0].set(xlabel="pasos", ylabel="s/imagen (±IC95%)", title="Tiempo vs pasos"); ax[1].set(xlabel="pasos", title="Diversidad (RMS por pares)")
for a in ax: a.grid(alpha=.3); a.legend()
plt.show()""")
code("""avail = [(r.scheduler, int(r.steps)) for r in S.itertuples()]
fig, axs = plt.subplots(len(avail), 1, figsize=(10, 1.4 * len(avail)))
for ax_, (s, n) in zip(axs, avail):
    ax_.imshow(core.make_grid([core.load_png(core.image_path(RESULTS, s, n, sd)) for sd in seeds[:8]], 8, scale=2, pad=1))
    ax_.set_ylabel(core.cfg_label(s, n), rotation=0, ha="right"); ax_.set_xticks([]); ax_.set_yticks([])
plt.tight_layout(); plt.show()""")
md("""## 16. Interpretación y demostración
Escriba aquí **al menos tres hallazgos interpretados** con base en *sus* tablas y gráficos (qué se midió, qué significa, qué limitación tiene). La demostración interactiva está en la app (`streamlit run app.py`): galería con selector de pasos/semilla y generación en vivo.

> **Recordatorio:** el modelo es incondicional; **no se genera desde una descripción textual**. Solo se controlan semilla, pasos y scheduler.

1. _Hallazgo 1:_
2. _Hallazgo 2:_
3. _Hallazgo 3:_

**Limitaciones:** dominio 32×32 tipo CIFAR-10; muestra pequeña (16 semillas); calidad subjetiva; sin FID/KID; tiempos válidos solo para el equipo registrado.""")
nb.cells = C
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "notebook_didactico.ipynb")
print("notebook escrito:", len(C), "celdas")
