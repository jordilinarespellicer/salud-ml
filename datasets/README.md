---
description: Datasets sintéticos clínicos del curso. No son datos de pacientes reales.
---

# Datasets sintéticos

{% hint style="warning" %}
**Todos los datos son sintéticos.** No representan ni derivan de pacientes reales. Se generan de forma reproducible (semilla fija) con [`00_Datos_Sinteticos_Maestro.ipynb`](https://colab.research.google.com/drive/1f_C8Y8JN1S4YF-EmA_IIKUnbq-tcg1Z_), y también de forma automática en la **primera celda de cada notebook**. El aviso ético es, además, contenido formativo: enseña por qué en salud no se trabaja con datos reales sin control.
{% endhint %}

Estos ficheros son el **hilo conductor** de todo el curso: se reutilizan unidad a unidad para que cada técnica se practique siempre sobre el mismo escenario clínico.

| Fichero | Contenido | Se usa en |
|---------|-----------|-----------|
| [`pacientes.csv`](https://drive.google.com/file/d/1Ku0j-sAf8Cr3FPT-DGm8v5p4h_2BmV5U/view?usp=drive_link) | Cohorte de 20 000 pacientes: demografía, constantes y analítica → **riesgo cardiovascular** (clasificación `evento_cv` y regresión `riesgo_cv_10a`) | U2–U6 |
| [`pacientes_sucio.csv`](https://drive.google.com/file/d/1geOuVmKqhBvBf52NyVFBvpxw7Ypbucp2/view?usp=drive_link) | La misma tabla con **problemas de calidad** inyectados a propósito (unidades mezcladas, nulos, outliers, duplicados, texto en campos numéricos) | U2 (EDA y limpieza) |
| [`urgencias_diarias.csv`](https://drive.google.com/file/d/1EpQ9Lcb-f-iDqBOA3f3sT_pGLBp2G56u/view?usp=drive_link) | **Serie temporal** de ingresos diarios en urgencias (2 años) con estacionalidad y temporada de gripe | U7 |
| [`notas_clinicas.csv`](https://drive.google.com/file/d/1cWvZFsNd1d-Wd_B8G2eTLewqyjiydE0x/view?usp=drive_link) | Notas clínicas cortas sintéticas con **especialidad** y **prioridad** (etiquetas) | U4 (texto), U9 (NER) |
| [`centros.csv`](https://drive.google.com/file/d/1rxxkSTg-hsyiLlC6ppGpKAoMjmBrolM-/view?usp=drive_link) | Catálogo de centros/hospitales (tipo, área, camas, servicios) | U6 (clustering) |
| [`wearable.csv`](https://drive.google.com/file/d/1az7oq8Rzkts0u37ijWVaRTvUnmpbNU7o/view?usp=drive_link) | Señal sencilla tipo wearable (frecuencia cardiaca, pasos, sueño) para un ejemplo de señal y anomalías | U7 / U8 |

## Esquemas por columna

### `pacientes.csv` (20 000 filas)

| Columna | Tipo | Descripción |
|---|---|---|
| `paciente_id` | str | Identificador `P00001`… |
| `edad` | int | 18–95 (cohorte sesgada a adultos mayores) |
| `sexo` | cat | `M` / `F` |
| `imc` | float | Índice de masa corporal |
| `ta_sistolica` | int | Tensión arterial sistólica (mmHg) |
| `ta_diastolica` | int | Tensión arterial diastólica (mmHg) |
| `glucemia` | float | Glucosa en ayunas (mg/dL) |
| `colesterol_total` | float | Colesterol total (mg/dL) |
| `hdl` | float | Colesterol HDL (mg/dL) — protector |
| `tabaquismo` | cat | `nunca` / `ex` / `activo` |
| `actividad_fisica` | cat | `baja` / `media` / `alta` |
| `antecedentes_familiares` | int | 0/1 |
| `diabetes` | int | 0/1 (glucemia > 126 mg/dL) — etiqueta alternativa |
| `riesgo_cv_10a` | float | **Riesgo cardiovascular a 10 años (%)** — objetivo de **regresión** |
| `evento_cv` | int | 0/1 — objetivo de **clasificación** (prevalencia ≈ 19 %) |

Las relaciones son **realistas e interpretables** (edad, tensión, tabaquismo, glucemia y colesterol suman riesgo; HDL y actividad física lo bajan), de modo que los coeficientes de la regresión logística y los valores SHAP cuentan una historia clínica creíble.

### `urgencias_diarias.csv` (730 filas)

`fecha` · `ingresos` (conteo) · `festivo` (0/1) · `temporada_gripe` (0/1) · `temperatura` (°C). Patrones incorporados: estacionalidad **semanal** (pico lunes/fin de semana), **anual** (invierno > verano), **temporada de gripe** y **festivos**.

### `notas_clinicas.csv` (3 000 filas)

`texto` (nota breve sintética) · `especialidad` (`cardiología`/`respiratorio`/`digestivo`/`neurología`/`traumatología`) · `prioridad` (`alta`/`media`/`baja`) · `centro_id`. Texto **plausible pero genérico**, sin datos identificativos.

### `centros.csv` (24 filas)

`centro_id` · `tipo` (`hospital`/`centro de salud`) · `area` · `camas` · `n_servicios` · `urgencias_dia_media` · `ratio_mayores65`.

### `wearable.csv` (200 sujetos × 30 días)

`sujeto_id` · `dia` · `fc_reposo` (lpm) · `pasos` · `horas_sueno`. Incluye algunos sujetos con **episodios anómalos** para practicar detección de anomalías.

### Datasets públicos (imagen y señal)

Para imagen y señal médica **no** sintetizamos: usamos recursos públicos, ligeros y estándar, que se descargan en el propio notebook.

| Recurso | Uso | Enlace |
|---|---|---|
| **MedMNIST v2** (PneumoniaMNIST, DermaMNIST, BloodMNIST, RetinaMNIST, BreastMNIST) | CNN y ViT en U8 | [medmnist.com](https://medmnist.com/) · [HF](https://huggingface.co/datasets/albertvillanova/medmnist-v2) |
| `dima806/chest_xray_pneumonia_detection` | Modelo ViT ya afinado, `pipeline()` en U9 | [HF](https://huggingface.co/dima806/chest_xray_pneumonia_detection) |
| `d4data/biomedical-ner-all` | NER biomédico (EN) en U9 | [HF](https://huggingface.co/d4data/biomedical-ner-all) |
| `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es` | Modelo clínico **en español** en U9 | [HF](https://huggingface.co/PlanTL-GOB-ES/roberta-base-biomedical-clinical-es) |

## Cómo regenerarlos

Tienes dos vías, ambas reproducibles:

* **En Colab:** abre [`00_Datos_Sinteticos_Maestro.ipynb`](https://colab.research.google.com/drive/1f_C8Y8JN1S4YF-EmA_IIKUnbq-tcg1Z_) y ejecuta *Entorno de ejecución → Ejecutar todo*. El código está comentado para que puedas **adaptarlos** (escala, patrones, tipos de "suciedad").
* **En local:** ejecuta `python gen_datasets_salud.py` (en esta carpeta). Es el mismo generador, como script de respaldo.
