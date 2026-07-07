---
description: >-
  Guía para impartir el curso: reparto de las 3 sesiones síncronas (2 h cada
  una), qué queda en asíncrono, fronteras de coordinación con el resto del
  programa y plan de vídeos con guiones. Documento de uso docente.
---

# Guía docente y plan de vídeos

> **Para el profesorado.** Este documento organiza la impartición del hilo de **Machine Learning** dentro del *Curso de Especialización de IA aplicada a la Salud* (marco ValgrAI). No es material del alumnado: es tu mapa para las clases y las grabaciones.

El curso se imparte en **3 sesiones online de 2 horas** más **material asíncrono** (lectura en GitBook + notebooks de Colab + vídeos). La regla de oro con este público —profesionales sanitarios, en su mayoría no programadores— es que **lo síncrono construya criterio y confianza**, y que **lo denso y ejecutable viva en el asíncrono**, que se puede repasar a ritmo propio.

## 1. Mapa del curso: síncrono vs. asíncrono

| Unidad | Cómo se imparte |
|---|---|
| U0 · Puesta a punto (Colab y Python) | **Asíncrono** (on-ramp opcional, antes de empezar) |
| U1 · ¿Qué pasa con la IA en salud? | **Síncrono** (Sesión 1) |
| U2 · Fundamentos, método 2026 y EDA | **Síncrono** (Sesión 1) + notebook |
| U3 · Evaluar bien: métricas y validación | **Síncrono** (Sesión 1) + notebook |
| U4 · Supervisados I (logística) | **Síncrono** (Sesión 2) + notebook |
| U5 · Supervisados II + cómo elegir | **Síncrono** (Sesión 2, pieza central) + notebook |
| U6 · No supervisado (fenotipado) | **Asíncrono** + notebook |
| U7 · Series temporales (urgencias) | **Asíncrono** + notebook (candidato a vídeo) |
| U8 · Redes, imagen y señal | **Síncrono** (Sesión 3) + notebook |
| U9 · Modelos fundacionales | **Síncrono** (Sesión 3) + notebooks |
| U10 · La IA como copiloto | **Síncrono** (Sesión 3, demo) + notebook (vídeo estrella) |
| U11 · Ética, sesgo y privacidad | **Asíncrono** (breve, transversal) |

## 2. Reparto de las tres sesiones (minutado)

Cada sesión dura 120 minutos e incluye un descanso corto. Los tiempos son orientativos; prioriza el **debate y las preguntas** sobre terminar el guion.

### Sesión 1 — Fundamentos y evaluar bien

| Tramo | Contenido | min |
|---|---|---|
| 1 | **U1** · La IA en salud hoy: posibilidades, límites y encuadre realista | 20 |
| 2 | **U2** · Fundamentos, método 2026 y **EDA en vivo** sobre `pacientes.csv` (abrir el notebook y mirar los datos) | 55 |
| — | Descanso | 5 |
| 3 | **U3** · Evaluar bien: sensibilidad/especificidad, VPP/VPN, ROC/PR y **calibración**, con lenguaje clínico | 35 |
| 4 | Cierre + qué practicar en asíncrono (U0 si hace falta, notebooks U2–U3) | 5 |

> 💡 **Objetivo de la sesión 1.** Que salgan convencidos de dos cosas: que **mirar y limpiar los datos** es la mitad del trabajo, y que **una métrica mal elegida engaña**. Si interiorizan esto, el resto del curso se sostiene.

### Sesión 2 — Modelos y cómo elegir el mejor (el corazón)

| Tramo | Contenido | min |
|---|---|---|
| 0 | Repaso relámpago de la sesión 1 | 5 |
| 1 | **U4** · Supervisados I: regresión lineal y, sobre todo, **logística** (coeficientes como *odds ratio*, el lenguaje del riesgo clínico) | 40 |
| — | Descanso | 5 |
| 2 | **U5** · Supervisados II + **validación cruzada + búsqueda de hiperparámetros + elegir el mejor modelo** + explicabilidad (SHAP). **Pieza central del curso** | 65 |
| 3 | Cierre + notebooks U4–U5 para practicar | 5 |

> 💡 **Objetivo de la sesión 2.** Que entiendan el método honesto de **comparar modelos y quedarse con el mejor sin autoengañarse**, y que vean que a veces **el modelo simple gana**. Es la competencia más transferible de todo el curso.

### Sesión 3 — Del dato a la IA que ya existe

| Tramo | Contenido | min |
|---|---|---|
| 0 | Repaso relámpago de la sesión 2 | 5 |
| 1 | **U8** · Redes, imagen y señal: intuición de CNN/ViT y **demo con imagen médica** (MedMNIST) | 40 |
| — | Descanso | 5 |
| 2 | **U9** · Modelos fundacionales: **Hugging Face** y **APIs (OpenRouter)** — demo de "usar un modelo ya entrenado" | 35 |
| 3 | **U10** · La IA como copiloto: **demo en vivo del bucle** (de `pacientes.csv` al mejor modelo, dirigiendo a un asistente) | 30 |
| 4 | Cierre del curso: cómo seguir (U6, U7, U11 en asíncrono) y recursos | 5 |

> 💡 **Objetivo de la sesión 3.** El mensaje que se llevan a casa: **"ya casi nadie entrena desde cero; hoy elegimos, integramos y evaluamos modelos ya hechos, y dirigimos a un asistente que escribe el código"**. Es lo que hace que este curso sea posible para quien no programa.

## 3. Cómo dar una sesión en directo (checklist)

Antes de cada sesión:

- Abre el **notebook** de la unidad y ejecuta la **primera celda** (genera los datos) para tenerlo caliente.
- Ten a mano el capítulo de GitBook correspondiente para compartir pantalla y **hacer scroll mientras explicas** (el texto está aireado y con cajas para eso).
- Si vas a hacer una demo con un asistente de IA (U10) o una API (U9), comprueba antes la conexión y la clave.

Durante la sesión, apóyate en el patrón del material: **primero la intuición y el "para qué" clínico, luego el código ya escrito** (no lo teclees en vivo salvo demo), y siempre **interpreta el resultado** en voz alta. Invita a que imaginen el ejemplo en su propio servicio.

## 4. Fronteras de coordinación (para no pisar ni repetir)

Tu parte es el hilo de **Machine Learning** dentro del Módulo I. Encaja así con el resto:

- **Con la introducción general a la IA (otro docente):** esa parte define IA fuerte/débil, áreas, etc. Tu **U1 no repite** eso: va directa a *posibilidades, límites y ejemplos reales en salud*.
- **Con la parte de PLN / modelos de lenguaje (equipo de la UA):** ellos cubren PLN y prompting **desde la lingüística/biomedicina**. Tú usas los LLM **como herramienta para hacer ciencia de datos** (analizar un dataset, proponer modelos). En **U9** usas un modelo clínico en español solo como **puente**, sin entrar en teoría de PLN.
- **Con Isabel Ferri y Carlos Aliaga (compartís las 20 h de ML):** esta propuesta asume que tú lideras el *"ML práctico de 2026 + salto a fundacionales + copiloto de IA"*. Si ellos cubren en profundidad la teoría de alguna técnica concreta, aligera ese punto para no solapar.
- **Con el Módulo II (radiómica, imagen avanzada):** tu **U8** da la **intuición general** de CNN/ViT con datasets de juguete; ellos entran en radiómica real. Preparas el terreno.
- **Con el Módulo III (historia clínica, aspectos legales):** tu **U11** es **transversal y breve**; el peso legal (protección de datos, marco regulatorio) lo lleva su sesión. Dilo explícitamente en clase.

## 5. Plan de vídeos asíncronos

Tu intuición de grabar vídeos —sobre todo recorridos de principio a fin— es lo más valioso para este público. Propuesta de catálogo en dos niveles.

### 5.1 Vídeos "concepto" (5–10 min)

Uno por unidad síncrona, como refuerzo de repaso: **U1, U2, U3, U4, U5, U8, U9, U10**. Formato: screencast del capítulo de GitBook con tu voz, subrayando las 3–4 ideas clave de cada unidad. Se pueden grabar deprisa leyendo la sección "Qué llevarte" de cada unidad y desarrollándola.

### 5.2 Vídeos "de principio a fin" (15–25 min, las piezas estrella)

Tres recorridos completos, cada uno consolidando un bloque del curso:

**Vídeo 1 · De un CSV clínico a un modelo evaluado y explicado** (consolida U2–U5)

- Base: notebook `U10_Copiloto_DataScience.ipynb` (o encadenando `U02`→`U05`).
- Guion: abrir `pacientes.csv` → EDA rápido (mirar y limpiar) → partición **sin fugas** → probar varios modelos → **validación cruzada** → **hiperparámetros** → **elegir el mejor** por AUC/calibración → **interpretar con SHAP/importancias** qué sube el riesgo de un paciente. Mensaje: criterio clínico + método honesto.

**Vídeo 2 · Clasificar imagen médica con un modelo fundacional en 15 minutos** (consolida U8–U9)

- Base: notebook `U08_Redes_Imagen.ipynb` + `U09a_Fundacionales_HF.ipynb`.
- Guion: qué es una CNN/ViT en intuición → cargar **MedMNIST** (PneumoniaMNIST) → usar un modelo **ya afinado** de Hugging Face con `pipeline()` **sin entrenar desde cero** → interpretar aciertos/errores y avisar de la validación externa. Mensaje: hoy somos **usuarios** de modelos entrenados.

**Vídeo 3 · Usar Claude / Qwen para analizar un dataset y encontrar el mejor modelo** (consolida U10)

- Base: notebook `U10_Copiloto_DataScience.ipynb` + guía `../estudio/prompting-loops.md`.
- Guion: plantear el problema a un asistente (prompt con contexto + objetivo + métrica clínica) → dejar que proponga y ejecute → **leer, criticar y pedir la siguiente vuelta** (loop) → validar como humano al mando. Mostrar de pasada qué son **Cowork, Claude Code, Codex, Manus**. Mensaje: no programas, **diriges un bucle**.

### 5.3 Recomendación de formato

Screencast de Colab/GitBook con tu voz (webcam en esquina, opcional). Guion breve por vídeo (te lo puedo preparar a partir de cada notebook, para que grabar sea "leer y hacer"). Los notebooks ya están subidos a la **carpeta de Drive del curso** y cada unidad enlaza con **"Abrir en Colab"** al cuaderno correspondiente.

## 6. Prerrequisitos y recomendación para el alumnado

- Antes de la Sesión 1, recomendar (opcional) la **U0** a quien no haya tocado nunca Colab.
- Recordar en cada sesión que **no hay que saber programar**: el código está escrito y comentado, y el asistente puede generarlo.
- Insistir en el **aviso de datos sintéticos**: todo el curso usa datos inventados de forma controlada; es también una lección sobre por qué en salud no se trabaja con datos reales sin garantías.
