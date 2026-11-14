# Grupo 10 · Evaluación de generación de imágenes con un DDPM preentrenado

Aplicativo Streamlit + código reproducible para el trabajo *Modelo de difusión (DDPM)* de **Sistemas Inteligentes y Machine Learning**.
Modelo analizado: [`google/ddpm-cifar10-32`](https://huggingface.co/google/ddpm-cifar10-32) (incondicional, imágenes 32×32). **No se entrena nada.**

> **Resultados incluidos.** `results/` contiene la ejecución **real** del protocolo (8 configuraciones × 16 semillas = 128 imágenes, en CPU) con los pesos oficiales
> suministrados por el grupo. Se ejecutó una sola vez; los tiempos valen para la máquina registrada en `results/env.json`.
> **Pendiente (requiere personas):** calificaciones visuales a ciegas (pestaña *Calificación a ciegas*), selección de configuración, prueba final y la interpretación escrita.
> **Pesos del modelo:** no van en este ZIP (143 MB). Ejecute `python preparar_modelo.py RUTA_DE_SUS_ARCHIVOS_DESCOMPRIMIDOS` y creará `model/ddpm-cifar10-32/` (U-Net en 2 fragmentos de <100 MB, para subirlos a GitHub sin Git LFS). La app usa **siempre primero ese modelo local**. Si la carpeta no está, la pestaña *Generación en vivo* pide **subir los archivos del modelo** (o descargarlo de Hugging Face).

## Contenido
| Archivo | Función |
|---|---|
| `app.py` | App Streamlit (10 pestañas: teoría, difusión directa, galería, evaluación, calificación a ciegas, generación en vivo, ejecutar, 16 etapas, limitaciones/sustentación) |
| `ddpm_core.py` | Núcleo: carga del modelo, muestreo explícito U-Net + scheduler (DDPM/DDIM), tiempos, memoria, métricas, selección |
| `theory.py` | Matemática de la difusión (NumPy) y diagrama original de la U-Net |
| `content.py` | Textos de las 16 etapas, preguntas de sustentación, limitaciones, referencias APA 7 |
| `run_experiments.py` | Ejecuta todo el protocolo (reanudable) |
| `preparar_modelo.py` | Ordena y parte los pesos del modelo en `model/ddpm-cifar10-32/` |
| `notebook_didactico.ipynb` / `make_notebook.py` | Notebook con las 16 etapas (Markdown + código) |
| `requirements.txt`, `requirements-dev.txt`, `requirements.lock.txt` | Dependencias de la app/deploy, extras para el notebook y versiones con que se probó |

## Pasos de reproducción
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_experiments.py            # 8 configs x 16 semillas; reanudable. Descarga el modelo la primera vez.
streamlit run app.py                 # abrir http://localhost:8501
```
Opciones útiles: `--device cpu|cuda`, `--seeds 10`, `--model-path ./ddpm-cifar10-32 --offline` (si descargó el repositorio a mano con
`huggingface-cli download google/ddpm-cifar10-32 --local-dir ddpm-cifar10-32`).
También puede lanzar todo desde la pestaña **Ejecutar experimentos** de la app. En **Google Colab** (GPU T4): suba esta carpeta, `!pip install -r requirements.txt`, `!python run_experiments.py`, y descargue `results/`.

Flujo recomendado: (1) `run_experiments.py` → (2) abrir la app → (3) pestaña **Calificación a ciegas** (≥ 2 evaluadores) → (4) pestaña **Evaluación** (selección de configuración) → (5) **Ejecutar experimentos → Prueba final** → (6) redactar la interpretación en la app y en el informe.

## Diseño metodológico (qué se hizo y por qué)
* **Muestreo escrito explícitamente** (`sample_one`): U-Net → `scheduler.step`, en vez de `DDPMPipeline.__call__`. Permite medir tiempos limpios, capturar x_t / x̂₀ y demostrar comprensión.
* **Reproducibilidad:** ruido inicial desde `torch.Generator("cpu").manual_seed(semilla)` (idéntico en CPU y GPU); el mismo generador alimenta el ruido por paso de DDPM; DDIM con η = 0. `results/reproducibility.json` comprueba que misma semilla ⇒ misma imagen y otra semilla ⇒ otra imagen. Versiones y hardware en `results/env.json`; hash de revisión del checkpoint y model card en `results/`.
* **Conjuntos de semillas (declarados antes de ejecutar):** exploración {0–3} (warm-up/humo), **validación** {100–115} (comparar y seleccionar), **prueba final** {200–209} (evaluar una vez la configuración elegida vs línea base; nunca interviene en la selección).
* **Rejilla:** DDPM y DDIM × {25, 50, 100, 250} pasos (experimentos A–D de la guía). Línea base: **DDPM-100**.
* **Tiempo:** bucle completo (U-Net + scheduler) por imagen, batch 1, con warm-up descartado y sincronización CUDA. Memoria: máximo asignado por PyTorch (GPU) o RSS del proceso (CPU; su pico es monótono, se advierte en la app).
* **Métricas** (todas calculadas desde las imágenes guardadas): s/imagen (media, desv., IC95 %), ms/paso, diversidad (distancia RMS por pares, desv. por píxel), indicadores de ruido residual (variación total, contraste), PSNR/SSIM frente a la línea base con la misma semilla (**consistencia, no calidad**) y **calificación visual a ciegas 1–5** por evaluadores.
* **Selección pre-declarada:** entre las configuraciones con calificación media ≥ mejor − 0.5, la de menor tiempo. Sin calificaciones, la app **no selecciona**.
* **FID/KID no se calculan:** 10–16 muestras no bastan para estimarlos de forma válida (se explica en la app).

## Verificación hecha
Ejecución completa con los pesos oficiales: U-Net de **35 746 307 parámetros** (136.4 MB en float32), configuración idéntica a la de referencia, scheduler `DDPMScheduler` (β lineal 1e-4→0.02, T=1000, `clip_sample=True`, `variance_type=fixed_large`).
Reproducibilidad comprobada (`results/reproducibility.json`): misma semilla ⇒ imagen idéntica (diferencia máxima 0) en DDPM y DDIM; otra semilla ⇒ imagen distinta. App y notebook probados de punta a punta.

## Subir a GitHub y desplegar en Streamlit Community Cloud
1. Cree un repositorio y suba **todo el contenido de esta carpeta**, incluida `results/` y `model/` (≈140 MB en total; ningún archivo supera los 100 MB de GitHub):
   ```bash
   git init && git add . && git commit -m "App DDPM Grupo 10"
   git branch -M main && git remote add origin https://github.com/USUARIO/REPO.git && git push -u origin main
   ```
2. En https://share.streamlit.io → *Create app* → elija el repositorio, rama `main`, archivo principal **`app.py`**. En *Advanced settings* elija **Python 3.12**.
3. Al desplegar, la app instala `requirements.txt` (torch en versión CPU) y muestra los resultados precalculados. *Generación en vivo* usa el modelo de `model/` si está en el repositorio. **Si no lo subió a GitHub**, la app lo pedirá: elija *Subir archivos del modelo* y suba `model_index.json`, `scheduler_config.json`, `config.json` y `diffusion_pytorch_model.safetensors` (o un .zip que los contenga; límite de subida configurado: 250 MB), o *Descargar de Hugging Face*.
4. **Disco temporal:** en la nube lo que se escriba (calificaciones, interpretación) se pierde al reiniciar. Use los botones de descarga de la app y suba `results/ratings.csv` / `results/interpretacion.md` al repositorio para conservarlos. Todos los visitantes comparten el mismo `ratings.csv`.
5. No ejecute el protocolo completo en la nube (CPU compartida, horas de cómputo): hágalo en local y suba `results/`.
Si solo quiere la galería y las tablas (sin generación en vivo), puede quitar `torch`, `diffusers`, `huggingface_hub` y `safetensors` de `requirements.txt`: la app lo detecta y desactiva esas pestañas.

## Solución de problemas
* **`numpy>=1.17,<2.0 is required ... found numpy==2.x` al cargar el modelo (local):** un paquete antiguo del entorno (normalmente `transformers`) no soporta numpy 2. Solución recomendada: entorno limpio — `python -m venv .venv`, activarlo, `pip install -r requirements.txt`. Alternativa: `pip install -U transformers peft accelerate`.
* **Sin internet / Hugging Face bloqueado:** descargue el repositorio a mano y pase la carpeta (`--model-path`, o la caja *Modelo* de la app con *Solo archivos locales*). La carpeta debe tener `model_index.json`, `unet/` y `scheduler/`.
* **Tiempos distintos a los de `results/`:** son de la CPU registrada en `results/env.json`; en otra máquina cambian.

## Licencias y créditos
Modelo: Google / Hugging Face (revisar la licencia en la model card, guardada en `results/model_card.md`). Método: Ho et al. (2020); DDIM: Song et al. (2021). Código: `diffusers` (Apache-2.0). Referencias completas en APA 7 dentro de la app (pestaña *Limitaciones y sustentación*).
