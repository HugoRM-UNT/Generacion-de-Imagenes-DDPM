"""Textos del trabajo (etapas, sustentación, limitaciones, referencias). Sin lógica de cómputo."""

STAGES = [
 (1, "Definición del problema",
  "Describir la finalidad de generar imágenes y las restricciones de resolución y cómputo.",
  "Estudiar cómo un DDPM construye imágenes a partir de ruido y cómo el **número de pasos** y el **programador (scheduler)** "
  "cambian tiempo, memoria y calidad observable. Restricciones: imágenes RGB de **32×32** (dominio CIFAR-10), "
  "modelo **preentrenado y congelado** (no se entrena), cómputo modesto (CPU o GPU pequeña), evaluación de **calidad visual y costo**, "
  "no de exactitud de clasificación."),
 (2, "Tipo de aprendizaje",
  "Formular generación incondicional por difusión (no clasificación).",
  "Aprendizaje **no supervisado generativo** e **incondicional**: el modelo aproxima p(x) de CIFAR-10 sin etiquetas ni texto. "
  "Durante su entrenamiento original la tarea auxiliar es **regresión del ruido** (ε̂ ≈ ε, error cuadrático). "
  "Aquí solo hay **inferencia**: muestreo x_T → x_0. No hay clases, no hay accuracy."),
 (3, "Obtención y trazabilidad de datos",
  "Descargar el checkpoint de Hugging Face y revisar licencia/model card.",
  "Checkpoint oficial `google/ddpm-cifar10-32` (formato Diffusers). `run_experiments.py` registra el **hash de commit** descargado, "
  "guarda la **model card** en `results/model_card.md` y la etiqueta de licencia en `results/model_info.json`. "
  "Créditos: Ho, Jain & Abbeel (2020) para el método; Hugging Face/Diffusers para el código; CIFAR-10 (Krizhevsky, 2009) solo como referencia de dominio. "
  "La licencia debe confirmarse en la model card (aparece en el archivo guardado)."),
 (4, "Entradas, salidas y objetivo",
  "Identificar ruido inicial, timesteps, scheduler e imagen de salida.",
  "**Entradas:** ruido x_T ∼ N(0, I) de forma 3×32×32 (fijado por la semilla), timestep t ∈ {0,…,999}. "
  "**Red:** U-Net que devuelve ε̂_θ(x_t, t) (3×32×32). **Scheduler:** convierte (x_t, ε̂) en x_{t'}. "
  "**Salida:** imagen RGB 32×32 en [−1, 1] → uint8. **Objetivo de generación:** muestras plausibles y diversas al menor costo."),
 (5, "Análisis exploratorio",
  "Observar muestras de referencia del modelo y registrar su distribución visual sin extrapolar a datos reales.",
  "Se observan las muestras de la **línea base (DDPM-100)** y se resumen sus estadísticos de píxel (histograma por canal, contraste). "
  "Es una descripción **de lo que produce el modelo**, no de CIFAR-10 real (no se descarga el dataset); no se concluye nada sobre la distribución real."),
 (6, "Separación experimental",
  "Reservar semillas de prueba/comparación y congelar el modelo; no hay train local.",
  "Tres conjuntos de semillas **declarados antes de ejecutar**: **exploración** {0–3} (warm-up y pruebas de humo, nunca reportadas), "
  "**validación** {100–115} (comparar configuraciones y seleccionar) y **prueba final** {200–209} (solo para evaluar *una vez* la configuración elegida frente a la línea base). "
  "El modelo se pone en `eval()` con `requires_grad=False`."),
 (7, "Preprocesamiento / entorno",
  "Configurar Diffusers, PyTorch y dispositivo; registrar versiones, memoria y CPU/GPU.",
  "No hay preprocesamiento de datos (el modelo trabaja en [−1,1] y la conversión a uint8 es la única transformación de salida). "
  "Se registran versiones de bibliotecas, CPU/GPU, hilos, RAM y memoria de pesos en `results/env.json` y `results/model_info.json`."),
 (8, "Modelo de referencia",
  "Generar una muestra base con DDPM y 100 pasos.",
  "Configuración **C: DDPM, 100 pasos**, semillas de validación. Es la **línea base** contra la que se comparan tiempo, similitud y calidad de todas las demás."),
 (9, "Diseño de arquitectura",
  "Explicar la U-Net interna como estimador del ruido y el papel del scheduler.",
  "U-Net (encoder–decoder con conexiones de salto) de 4 niveles (128-256-256-256 canales), 2 ResNets por nivel en bajada y 3 en subida, "
  "autoatención a 16×16 y en el bloque central, y embedding sinusoidal del timestep sumado en cada ResNet. "
  "El **scheduler** no tiene parámetros entrenables: aplica la fórmula del paso inverso usando ε̂ (ver pestaña *Teoría y arquitectura*)."),
 (10, "Configuración de inferencia",
  "Definir controles de pasos, semilla y número de imágenes; fijar resolución 32×32.",
  "Controles: `scheduler ∈ {DDPM, DDIM}`, `pasos`, `semilla`, `n imágenes`. Resolución fija 32×32, batch = 1 para medir latencia por muestra, "
  "DDIM con η = 0 (determinista), espaciado de timesteps `leading` de Diffusers."),
 (11, "Plan de hiperparámetros",
  "Comparar pasos {25, 50, 100, 250} con scheduler compatible; documentar valores no soportados.",
  "Rejilla completa: DDPM y DDIM × {25, 50, 100, 250} = 8 configuraciones (cubre los experimentos A, B, C y D de la guía). "
  "La compatibilidad se verifica construyendo cada scheduler y sus timesteps; si un valor falla, se documenta y se ajusta."),
 (12, "Ejecución y seguimiento",
  "Guardar imágenes y tiempo por muestra usando las mismas semillas.",
  "`run_experiments.py` guarda cada imagen PNG sin pérdida (`results/images/<config>/seed_<n>.png`) y una fila en `results/timings.csv` "
  "(scheduler, pasos, semilla, segundos, memoria, dispositivo, fecha). Es reanudable. Tiempo = bucle completo U-Net + scheduler, con warm-up previo descartado."),
 (13, "Validación",
  "Evaluar con un protocolo repetible: 10–20 semillas por configuración.",
  "16 semillas × 8 configuraciones = 128 imágenes. Se comprueba además la **reproducibilidad** (misma semilla ⇒ misma imagen) en `results/reproducibility.json`. "
  "Métricas: segundos/imagen (media, desv., IC95 %), memoria, diversidad (distancia RMS por pares, desv. por píxel), indicadores de ruido residual, "
  "similitud con la línea base (PSNR, SSIM) y **calificación visual a ciegas** por evaluadores."),
 (14, "Selección de configuración",
  "Elegir equilibrando calidad observada, diversidad y tiempo.",
  "Regla **pre-declarada**: entre las configuraciones cuya calificación media esté a menos de 0.5 puntos de la mejor, se elige la de **menor tiempo**. "
  "Si no hay calificaciones humanas, la app **no elige** (no se inventa calidad)."),
 (15, "Evaluación final",
  "Presentar tabla de latencia, memoria y juicio visual; FID opcional.",
  "Tabla de latencia/memoria/diversidad/similitud en la pestaña *Evaluación*, y evaluación final de *configuración elegida vs línea base* en las semillas de prueba {200–209}. "
  "**FID/KID no se calculan**: 10–16 imágenes son insuficientes para estimarlos de forma fiable (requieren miles de muestras y una red de características)."),
 (16, "Interpretación y demostración",
  "Implementar galería con selector de pasos/semilla y explicar que no se genera desde texto.",
  "Pestañas *Galería y demo* (selector de scheduler/pasos/semilla, mosaicos, proceso de denoising) y *Generación en vivo*. "
  "Aviso explícito: el modelo es **incondicional**; no recibe texto, así que **no existe prompt**; solo controlamos semilla, pasos y scheduler."),
]

QA = [
 ("1. ¿Qué ocurre en la difusión directa?",
  "Se corrompe una imagen x₀ añadiendo ruido gaussiano en T = 1000 pasos con una cadena de Markov fija (sin parámetros aprendidos): "
  "q(x_t | x_{t−1}) = N(√(1−β_t) x_{t−1}, β_t I). Gracias a la propiedad gaussiana se salta directamente a cualquier t: "
  "**x_t = √ᾱ_t · x₀ + √(1−ᾱ_t) · ε**, ε ∼ N(0, I), con ᾱ_t = ∏(1−β_s). Al final, ᾱ_T ≈ 0 y x_T es ruido casi puro. Esto se verifica numéricamente en la pestaña *Difusión directa*."),
 ("2. ¿Qué aprende a estimar la red durante entrenamiento?",
  "El **ruido ε** que se añadió: se toma x₀, un t al azar y un ε, se forma x_t y la U-Net minimiza ‖ε − ε̂_θ(x_t, t)‖². "
  "Es equivalente (por parametrización) a estimar la media del paso inverso o el score. Otras parametrizaciones estiman x₀ o v."),
 ("3. ¿Qué función cumple el scheduler?",
  "Define el **calendario de ruido** (β_t, ᾱ_t), **qué timesteps se visitan** (con 25 pasos se saltan de 40 en 40) y **la fórmula que convierte (x_t, ε̂) en x_{t'}**, "
  "incluyendo cuánto ruido nuevo se inyecta. No tiene pesos entrenables; cambiarlo no requiere reentrenar."),
 ("4. ¿Por qué CIFAR-10 no implica capacidad texto-a-imagen?",
  "Porque el modelo es **incondicional**: aprendió p(x) de 60 000 imágenes 32×32 de 10 clases sin ningún texto ni etiqueta de entrada. "
  "No existe un codificador de texto ni un mecanismo de condicionamiento (como cross-attention o guidance); la única entrada es ruido y t."),
 ("5. ¿Cuál es la diferencia entre DDPM y DDIM?",
  "DDPM muestrea con un proceso **estocástico** (inyecta ruido nuevo en cada paso). DDIM (Song et al., 2021) usa el mismo modelo entrenado pero una dinámica "
  "no markoviana con σ_t = η·(…): con η = 0 es **determinista** (misma x_T ⇒ misma imagen) y tolera **muy pocos pasos** con menor degradación. "
  "Con la misma red, cambian trayectoria, costo por calidad y la relación ruido→imagen."),
 ("6. ¿Por qué usar las mismas semillas al comparar?",
  "Para **controlar la variable de confusión**: la semilla fija x_T (y el ruido de DDPM). Así, las diferencias entre configuraciones se deben al scheduler/pasos y no a muestras distintas "
  "(diseño pareado). También habilita comparar imagen contra imagen y reproducir el experimento."),
 ("7. ¿Qué limitaciones tiene una evaluación visual con pocas muestras?",
  "Alta varianza (16 imágenes dan intervalos amplios), subjetividad del evaluador, sesgo de confirmación si no es a ciegas, escala ordinal poco sensible, "
  "resolución 32×32 que oculta defectos, y la calidad visual no mide cobertura de la distribución. FID/KID necesitan miles de muestras para ser fiables."),
 ("8. ¿Por qué no entrenar desde cero en esta práctica?",
  "Entrenar un DDPM de ~35.7 M de parámetros sobre CIFAR-10 exige del orden de cientos de miles de iteraciones en GPU (días). El objetivo didáctico es "
  "**entender la inferencia** y el compromiso pasos/calidad/tiempo, que se estudia igual con el checkpoint oficial y con reproducibilidad."),
]

LIMITATIONS = [
 "**Dominio estrecho:** solo genera imágenes 32×32 estilo CIFAR-10 (10 clases de objetos/animales). No sirve para otros dominios ni resoluciones; la salida no se debe usar como si fuera una foto real.",
 "**Sin control semántico:** al ser incondicional no se puede pedir 'un perro'. Solo se controlan semilla, pasos y scheduler.",
 "**Sesgos heredados:** reproduce la distribución, composición y sesgos de CIFAR-10 (clases, encuadres, estilos). No se auditó contra datos reales en este trabajo.",
 "**Muestra pequeña:** 16 semillas de validación y 10 de prueba: los intervalos de confianza son anchos; las diferencias pequeñas pueden ser ruido.",
 "**Calidad subjetiva:** la calificación visual depende de los evaluadores; sin FID/KID no hay medida distribucional.",
 "**Tiempos dependientes del equipo:** los segundos/imagen valen solo para la máquina registrada en `env.json`; no se extrapolan a otros equipos.",
 "**Memoria en CPU aproximada:** se reporta RSS del proceso (el pico es monótono); en GPU se reporta el máximo asignado por PyTorch.",
 "**DDPM con pocos pasos:** DDPM de Diffusers con timesteps espaciados es un muestreador *ad hoc* respecto del entrenamiento a 1000 pasos; se espera degradación frente a DDIM (hipótesis a contrastar con los datos propios, no una conclusión previa).",
 "**Métricas de apoyo no son calidad:** PSNR/SSIM vs línea base miden *consistencia con otra configuración*, no realismo; TV/contraste son indicadores de ruido residual.",
]

REFERENCES = [
 "Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017). GANs trained by a two time-scale update rule converge to a local Nash equilibrium. *Advances in Neural Information Processing Systems, 30*. https://arxiv.org/abs/1706.08500",
 "Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *Advances in Neural Information Processing Systems, 33*, 6840–6851. https://arxiv.org/abs/2006.11239",
 "Hugging Face. (s. f.). *DDPM pipeline*. Diffusers documentation. https://huggingface.co/docs/diffusers/api/pipelines/ddpm",
 "Google. (s. f.). *ddpm-cifar10-32* [Modelo]. Hugging Face. https://huggingface.co/google/ddpm-cifar10-32",
 "Krizhevsky, A. (2009). *Learning multiple layers of features from tiny images* [Informe técnico]. University of Toronto.",
 "Nichol, A. Q., & Dhariwal, P. (2021). Improved denoising diffusion probabilistic models. *Proceedings of the 38th International Conference on Machine Learning, 139*, 8162–8171. https://arxiv.org/abs/2102.09672",
 "Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. En N. Navab et al. (Eds.), *Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015* (pp. 234–241). Springer. https://doi.org/10.1007/978-3-319-24574-4_28",
 "Song, J., Meng, C., & Ermon, S. (2021). Denoising diffusion implicit models. *International Conference on Learning Representations*. https://arxiv.org/abs/2010.02502",
 "von Platen, P., Patil, S., Lozhkov, A., Cuenca, P., Lambert, N., Rasul, K., Davaadorj, M., & Wolf, T. (2022). *Diffusers: State-of-the-art diffusion models* [Software]. Hugging Face. https://github.com/huggingface/diffusers",
 "Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: From error visibility to structural similarity. *IEEE Transactions on Image Processing, 13*(4), 600–612. https://doi.org/10.1109/TIP.2003.819861",
]

RATING_SCALE = {
 1: "Ruido o manchas sin estructura",
 2: "Colores/formas globales pero sin objeto reconocible",
 3: "Objeto o escena difusa, con artefactos fuertes",
 4: "Reconocible, con defectos menores",
 5: "Nítida y coherente (parece una muestra de CIFAR-10)",
}
