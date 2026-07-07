---
description: Notebooks de Google Colab del curso. Datos sintéticos, generados en la primera celda.
---

# Notebooks de Colab

Cada unidad práctica tiene su notebook. **Todos son autosuficientes:** su **primera celda genera los datos sintéticos** si no están en la sesión, así que puedes abrir cualquiera y ejecutarlo directamente, sin descargar nada.

{% hint style="success" %}
Los notebooks también estarán en la **carpeta del curso en Google Drive**. Cada enlace *Abrir en Colab* se rellenará al subir los cuadernos a Drive. Mientras tanto, tienes los ficheros `.ipynb` en esta misma carpeta del repositorio.
{% endhint %}

## Índice

| Notebook | Unidad | Qué hace |
|----------|--------|----------|
| `00_Datos_Sinteticos_Maestro.ipynb` | — | Genera todos los datasets sintéticos (pacientes, urgencias, notas, centros, wearable) |
| `U00_Bienvenida_Colab.ipynb` | U0 | Primer contacto con Colab y Python: celdas, pandas y una gráfica |
| `U02_Fundamentos_EDA.ipynb` | U2 | EDA y limpieza sobre `pacientes_sucio.csv` con reglas de dominio clínico |
| `U03_Metricas_Validacion.ipynb` | U3 | Métricas clínicas, ROC/PR, calibración y coste del error |
| `U04_Supervisados_I.ipynb` | U4 | Lineal, **logística** (coeficientes como *odds ratio*) y Naïve Bayes sobre notas |
| `U05_Supervisados_II.ipynb` | U5 | SVM/RF/boosting + **validación cruzada + búsqueda de hiperparámetros + selección** + SHAP |
| `U06_No_Supervisado.ipynb` | U6 | Fenotipado de pacientes (clustering) + PCA + Isolation Forest |
| `U07_Series_Temporales.ipynb` | U7 | Urgencias diarias: descomposición, validación *walk-forward*, modelo con *features* |
| `U08_Redes_Imagen.ipynb` | U8 | MLP tabular → **CNN con MedMNIST** → **ViT / transfer learning** |
| `U09a_Fundacionales_HF.ipynb` | U9 | `pipeline()` de imagen biomédica + NER clínico (ES/EN) con Hugging Face |
| `U09b_APIs_OpenRouter.ipynb` | U9 | Llamar a OpenAI/Claude/Qwen vía OpenRouter para analizar un dataset |
| `U10_Copiloto_DataScience.ipynb` | U10 | Recorrido **end-to-end** guiado por prompting/loops |

> Los notebooks de **U8** se benefician de **GPU** (gratuita en Colab: *Entorno de ejecución → Cambiar tipo de entorno → GPU*).

{% hint style="info" %}
**Enlaces "Abrir en Colab" (pendientes de Drive).** Cuando subas los `.ipynb` a la carpeta de Google Drive del curso, sustituye cada `PENDIENTE_DRIVE` por el enlace de Colab correspondiente. También puedes servirlos directamente desde GitHub con `https://colab.research.google.com/github/<usuario>/ml-salud-2026/blob/main/notebooks/<archivo>.ipynb`.
{% endhint %}
