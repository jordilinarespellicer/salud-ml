---
description: Notebooks de Google Colab del curso. Datos sintéticos, generados en la primera celda.
---

# Notebooks de Colab

Cada unidad práctica tiene su notebook. **Todos son autosuficientes:** su **primera celda genera los datos sintéticos** si no están en la sesión, así que puedes abrir cualquiera y ejecutarlo directamente, sin descargar nada.

{% hint style="success" %}
Cada enlace **abre el notebook directamente en Google Colab**. Al abrirlo, haz *Archivo → Guardar una copia en Drive* para poder ejecutarlo y editarlo con tu cuenta.
{% endhint %}

## Índice

| Notebook | Unidad | Qué hace |
|----------|--------|----------|
| [`00_Datos_Sinteticos_Maestro`](https://colab.research.google.com/drive/1f_C8Y8JN1S4YF-EmA_IIKUnbq-tcg1Z_) | — | Genera todos los datasets sintéticos (pacientes, urgencias, notas, centros, wearable) |
| [`U00_Bienvenida_Colab`](https://colab.research.google.com/drive/1FOLj4cb1ZnwhrZ_retpo5i9wS3pKfteX) | U0 | Primer contacto con Colab y Python: celdas, pandas y una gráfica |
| [`U02_Fundamentos_EDA`](https://colab.research.google.com/drive/1PeU5pfqYGkJTG5U3hOB5AvxDZrAVwh9L) | U2 | EDA y limpieza sobre `pacientes_sucio.csv` con reglas de dominio clínico |
| [`U03_Metricas_Validacion`](https://colab.research.google.com/drive/1YG0YK4XwBZHaAcetkIJcUnkaOqdL15fM) | U3 | Métricas clínicas, ROC/PR, calibración y coste del error |
| [`U04_Supervisados_I`](https://colab.research.google.com/drive/1mpLBIam_ebKCo7P6PpXDJHnkJBrbhDRz) | U4 | Lineal, **logística** (coeficientes como *odds ratio*) y Naïve Bayes sobre notas |
| [`U05_Supervisados_II`](https://colab.research.google.com/drive/1P-4mHFE11kJX6zgT-Ajm4KQR4Y4_vjy3) | U5 | SVM/RF/boosting + **validación cruzada + búsqueda de hiperparámetros + selección** + SHAP |
| [`U06_No_Supervisado`](https://colab.research.google.com/drive/1JlWxl0hzVbrte3E4Z0FXQVfPfQVDYDid) | U6 | Fenotipado de pacientes (clustering) + PCA + Isolation Forest |
| [`U07_Series_Temporales`](https://colab.research.google.com/drive/1PWP-WLyNPgAEk4b4WmbdeqQn2r_bQ7co) | U7 | Urgencias diarias: descomposición, validación *walk-forward*, modelo con *features* |
| [`U08_Redes_Imagen`](https://colab.research.google.com/drive/1NGzKU1gh2CaN5Cd9sddN7mWqmpang_A_) | U8 | MLP tabular → **CNN con MedMNIST** → **ViT / transfer learning** |
| [`U09a_Fundacionales_HF`](https://colab.research.google.com/drive/1MgvsLL_6QYwjP1fPM7JWGEJKpfJWBCY8) | U9 | `pipeline()` de imagen biomédica + NER clínico (ES/EN) con Hugging Face |
| [`U09b_APIs_OpenRouter`](https://colab.research.google.com/drive/1MU72hahjS96Nf1I3RfXOsqJkHrSJq_OI) | U9 | Llamar a OpenAI/Claude/Qwen vía OpenRouter para analizar un dataset |
| [`U09c_TabFM_vs_XGBoost`](https://colab.research.google.com/drive/1_z5-OFwwgyrtAamOIj2_fUwmTVPK3Izn) | U9 | **TabFM** (modelo fundacional tabular, *zero-shot*) contra un **XGBoost** afinado, sobre `pacientes_sucio.csv` limpiado — **requiere GPU** |
| [`U10_Copiloto_DataScience`](https://colab.research.google.com/drive/1Nz_xdzwz0SX_fOEbzSl0jbVLVHtUnu5H) | U10 | Recorrido **end-to-end** guiado por prompting/loops |

> Los notebooks de **U8** se benefician de **GPU** (gratuita en Colab: *Entorno de ejecución → Cambiar tipo de entorno → GPU*).

> Los ficheros `.ipynb` también están en esta carpeta del repositorio como respaldo.
