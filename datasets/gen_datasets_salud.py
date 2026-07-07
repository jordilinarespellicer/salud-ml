#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador de datasets sintéticos clínicos — Curso de Machine Learning para Salud (ValgrAI).
TODOS LOS DATOS SON SINTÉTICOS. No representan ni derivan de pacientes reales.
Reproducible (semilla fija). Este mismo código vive en 00_Datos_Sinteticos_Maestro.ipynb.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(2026)   # semilla fija -> reproducible
OUT = Path(".")

# =====================================================================
# 1) CATÁLOGO DE CENTROS  (centros.csv)  — análogo a tiendas.csv
# =====================================================================
AREAS = ["Valencia", "Alicante", "Castellón", "Madrid", "Barcelona",
         "Sevilla", "Bilbao", "Zaragoza"]
n_centros = 24
tipo = RNG.choice(["hospital", "centro de salud"], n_centros, p=[0.35, 0.65])
camas = np.where(tipo == "hospital",
                 RNG.integers(120, 900, n_centros),
                 RNG.integers(0, 25, n_centros))
n_servicios = np.where(tipo == "hospital",
                       RNG.integers(12, 40, n_centros),
                       RNG.integers(3, 10, n_centros))
centros = pd.DataFrame({
    "centro_id": [f"C{i:03d}" for i in range(1, n_centros + 1)],
    "tipo": tipo,
    "area": RNG.choice(AREAS, n_centros),
    "camas": camas,
    "n_servicios": n_servicios,
    # variables de apoyo para clustering (perfil del centro)
    "urgencias_dia_media": np.where(tipo == "hospital",
                                    RNG.integers(80, 320, n_centros),
                                    RNG.integers(5, 40, n_centros)),
    "ratio_mayores65": RNG.uniform(0.12, 0.34, n_centros).round(3),
})
centros.to_csv(OUT / "centros.csv", index=False)

# =====================================================================
# 2) TABLA CENTRAL DE PACIENTES  (pacientes.csv)  — análoga a ventas_historico
#    Objetivo regresión: riesgo_cv_10a (%). Objetivo clasificación: evento_cv (0/1).
#    Etiqueta secundaria: diabetes (0/1).
#    Diseño: relaciones realistas e INTERPRETABLES para que la logística
#    y los valores SHAP cuenten una historia clínica creíble.
# =====================================================================
N = 20000

# --- Demografía ---
sexo = RNG.choice(["M", "F"], N, p=[0.49, 0.51])
# edad sesgada a adultos mayores (mezcla de dos normales truncadas)
edad = np.clip(np.where(RNG.random(N) < 0.6,
                        RNG.normal(58, 15, N),
                        RNG.normal(40, 12, N)), 18, 95).round().astype(int)

# --- Hábitos ---
# tabaquismo: prob de fumador activo baja con la edad (mayores lo han dejado)
p_activo = np.clip(0.28 - 0.0016 * (edad - 18), 0.06, 0.28)
p_ex = np.clip(0.10 + 0.004 * (edad - 18), 0.10, 0.40)
p_nunca = 1 - p_activo - p_ex
tabaquismo = np.array([RNG.choice(["nunca", "ex", "activo"], p=[pn, pe, pa])
                       for pn, pe, pa in zip(p_nunca, p_ex, p_activo)])
actividad = RNG.choice(["baja", "media", "alta"], N, p=[0.4, 0.4, 0.2])
antecedentes = RNG.binomial(1, 0.22, N)

# --- Antropometría y analítica (correladas con edad/hábitos) ---
act_num = pd.Series(actividad).map({"baja": 1.5, "media": 0.0, "alta": -1.3}).values
imc = np.clip(RNG.normal(26.5, 4.2, N) + 0.03 * (edad - 50) + act_num
              + RNG.normal(0, 1.0, N), 15, 48).round(1)

fuma_act = (tabaquismo == "activo").astype(float)
fuma_ex = (tabaquismo == "ex").astype(float)

ta_sist = np.clip(RNG.normal(120, 12, N) + 0.45 * (edad - 45) + 0.7 * (imc - 25)
                  + 6 * fuma_act + RNG.normal(0, 6, N), 85, 220).round().astype(int)
ta_diast = np.clip(0.55 * ta_sist + RNG.normal(20, 6, N), 50, 130).round().astype(int)

# glucemia: marcador de diabetes; sube con IMC, edad, antecedentes
glucemia = np.clip(RNG.normal(100, 12, N) + 1.0 * (imc - 25) + 0.3 * (edad - 45)
                   + 9 * antecedentes + RNG.normal(0, 7, N), 60, 320).round(1)
diabetes = (glucemia > 126).astype(int)   # criterio clínico simplificado (glucosa en ayunas)

colesterol = np.clip(RNG.normal(195, 30, N) + 0.4 * (edad - 45) + 1.1 * (imc - 25)
                     + RNG.normal(0, 15, N), 110, 380).round(1)
hdl = np.clip(RNG.normal(55, 12, N) - 0.25 * (imc - 25) + 6 * (sexo == "F")
              - 5 * fuma_act - 3 * act_num + RNG.normal(0, 6, N), 20, 110).round(1)

# --- Riesgo cardiovascular a 10 años (%): función NO lineal e interpretable ---
# log-odds aditivo (estilo score de riesgo), con interacciones suaves.
z = (-3.1
     + 0.062 * (edad - 50)
     + 0.019 * (ta_sist - 120)
     + 0.008 * (colesterol - 190)
     - 0.028 * (hdl - 55)
     + 0.055 * (imc - 25)
     + 0.85 * fuma_act + 0.30 * fuma_ex
     + 0.60 * diabetes
     + 0.45 * antecedentes
     + 0.55 * (sexo == "M")
     + 0.012 * np.maximum(edad - 65, 0) * fuma_act   # interacción edad×tabaco
     + RNG.normal(0, 0.35, N))
riesgo_p = 1 / (1 + np.exp(-z))                       # prob "verdadera" latente
riesgo_cv_10a = (100 * riesgo_p).round(1)             # objetivo de REGRESIÓN (%)
evento_cv = RNG.binomial(1, riesgo_p)                 # objetivo de CLASIFICACIÓN

pacientes = pd.DataFrame({
    "paciente_id": [f"P{i:05d}" for i in range(1, N + 1)],
    "edad": edad,
    "sexo": sexo,
    "imc": imc,
    "ta_sistolica": ta_sist,
    "ta_diastolica": ta_diast,
    "glucemia": glucemia,
    "colesterol_total": colesterol,
    "hdl": hdl,
    "tabaquismo": tabaquismo,
    "actividad_fisica": actividad,
    "antecedentes_familiares": antecedentes,
    "diabetes": diabetes,
    "riesgo_cv_10a": riesgo_cv_10a,
    "evento_cv": evento_cv,
})
pacientes.to_csv(OUT / "pacientes.csv", index=False)

# =====================================================================
# 3) VERSIÓN "SUCIA"  (pacientes_sucio.csv)  — para EDA y limpieza (U2)
#    Inyectamos a propósito los problemas de un volcado real.
# =====================================================================
d = pacientes.copy()

# 1) categorías inconsistentes de sexo
mapa_sexo = {"M": ["M", "m", "Masculino", "H", "Hombre"],
             "F": ["F", "f", "Femenino", "Mujer"]}
d["sexo"] = d["sexo"].map(lambda s: RNG.choice(mapa_sexo[s]))

# 2) unidades mezcladas: parte de la glucemia en mmol/L (dividir ~18) como texto con coma
idx = RNG.choice(d.index, int(N * 0.06), replace=False)
d["glucemia"] = d["glucemia"].astype(object)   # permite mezclar texto y números
d.loc[idx, "glucemia"] = [str(round(v / 18.0, 1)).replace(".", ",")
                          for v in d.loc[idx, "glucemia"]]

# 3) valores ausentes NO aleatorios (más nulos de HDL en jóvenes: menos analíticas)
prob_na = np.where(pacientes["edad"] < 40, 0.12, 0.04)
mask = RNG.random(N) < prob_na
d.loc[mask, "hdl"] = np.nan
idx = RNG.choice(d.index, int(N * 0.05), replace=False)
d.loc[idx, "colesterol_total"] = np.nan

# 4) outliers imposibles
idx = RNG.choice(d.index, 25, replace=False); d.loc[idx, "edad"] = RNG.integers(150, 260, 25)
idx = RNG.choice(d.index, 20, replace=False); d.loc[idx, "ta_sistolica"] = -RNG.integers(1, 90, 20)
idx = RNG.choice(d.index, 15, replace=False); d.loc[idx, "imc"] = RNG.uniform(80, 200, 15).round(1)

# 5) texto en campos numéricos (casteamos a object para mezclar texto y números)
d["colesterol_total"] = d["colesterol_total"].astype(object)
d["ta_diastolica"] = d["ta_diastolica"].astype(object)
idx = RNG.choice(d.index, 30, replace=False); d.loc[idx, "colesterol_total"] = "desconocido"
idx = RNG.choice(d.index, 20, replace=False); d.loc[idx, "ta_diastolica"] = "N/D"

# 6) categorías de tabaquismo inconsistentes
mapa_tab = {"nunca": ["nunca", "No fumador", "NUNCA"],
            "ex": ["ex", "exfumador", "Ex-fumador"],
            "activo": ["activo", "Fumador", "SI"]}
d["tabaquismo"] = d["tabaquismo"].map(lambda s: RNG.choice(mapa_tab[s]))

# 7) duplicados y casi-duplicados
d = pd.concat([d, d.sample(400, random_state=3)], ignore_index=True)

# 8) espacios sobrantes en el id
idx = RNG.choice(d.index, 200, replace=False)
d.loc[idx, "paciente_id"] = d.loc[idx, "paciente_id"].map(lambda s: " " + s + " ")

d = d.sample(frac=1, random_state=11).reset_index(drop=True)
d.to_csv(OUT / "pacientes_sucio.csv", index=False)

# =====================================================================
# 4) SERIE TEMPORAL DE URGENCIAS  (urgencias_diarias.csv)  — para U7
#    Ingresos diarios (conteo Poisson) con estacionalidad semanal + anual +
#    temporada de gripe + festivos.
# =====================================================================
N_DIAS = 730
fechas = pd.date_range("2024-01-01", periods=N_DIAS, freq="D")
doy = fechas.dayofyear.values
dow = fechas.dayofweek.values

# festivos aproximados (España): Año Nuevo, Reyes, 1 May, 15 Ago, 12 Oct, 1 Nov, 6/8/25 Dic
festivos_set = {(1, 1), (1, 6), (5, 1), (8, 15), (10, 12), (11, 1), (12, 6), (12, 8), (12, 25)}
festivo = np.array([1 if (f.month, f.day) in festivos_set else 0 for f in fechas])

# temporada de gripe: dic-feb (meses 12,1,2)
temporada_gripe = np.array([1 if f.month in (12, 1, 2) else 0 for f in fechas])

# temperatura media diaria (patrón anual + ruido)
temperatura = (15 + 10 * np.sin(2 * np.pi * (doy - 110) / 365) + RNG.normal(0, 2.5, N_DIAS)).round(1)

base = 120.0
estacional_anual = 1 + 0.14 * np.sin(2 * np.pi * (doy - 20) / 365)   # más en invierno
efecto_semana = np.where(dow == 0, 1.18,                              # pico lunes
                         np.where(dow >= 5, 1.10, 1.0))               # y fin de semana
efecto_gripe = 1 + 0.22 * temporada_gripe
efecto_festivo = 1 + 0.15 * festivo
mu = base * estacional_anual * efecto_semana * efecto_gripe * efecto_festivo
ingresos = RNG.poisson(np.maximum(mu, 1)).astype(int)

urgencias = pd.DataFrame({
    "fecha": fechas.strftime("%Y-%m-%d"),
    "ingresos": ingresos,
    "festivo": festivo,
    "temporada_gripe": temporada_gripe,
    "temperatura": temperatura,
})
urgencias.to_csv(OUT / "urgencias_diarias.csv", index=False)

# =====================================================================
# 5) NOTAS CLÍNICAS  (notas_clinicas.csv)  — texto libre para U4/U9
#    Texto sintético PLAUSIBLE PERO GENÉRICO, sin datos identificativos.
# =====================================================================
plantillas = {
    "cardiología": [
        "dolor torácico opresivo de {t} de evolución, irradiado a brazo izquierdo",
        "disnea de esfuerzo progresiva y palpitaciones, antecedente de hipertensión",
        "control de anticoagulación, fibrilación auricular conocida, sin dolor actual",
        "edemas en miembros inferiores y ortopnea, sospecha de insuficiencia cardiaca",
    ],
    "respiratorio": [
        "tos productiva de {t}, fiebre y disnea, auscultación con crepitantes",
        "sibilancias y opresión torácica, antecedente de asma, mala respuesta a inhalador",
        "disnea súbita y dolor pleurítico, saturación disminuida",
        "tos seca persistente y febrícula, contacto con caso respiratorio",
    ],
    "digestivo": [
        "dolor abdominal en fosa iliaca derecha de {t}, náuseas y febrícula",
        "epigastralgia y pirosis, relación con las comidas, sin signos de alarma",
        "diarrea de {t} sin productos patológicos, buen estado general",
        "ictericia y coluria, dolor en hipocondrio derecho",
    ],
    "neurología": [
        "cefalea intensa de inicio súbito, la peor de su vida, con fotofobia",
        "focalidad neurológica de {t}, debilidad en hemicuerpo y disartria",
        "mareo con giro de objetos y vómitos, sin cortejo vegetativo",
        "crisis comicial presenciada, recuperación progresiva del nivel de conciencia",
    ],
    "traumatología": [
        "caída casual con dolor e impotencia funcional en muñeca, deformidad",
        "lumbalgia mecánica de {t} tras esfuerzo, sin déficit neurológico",
        "esguince de tobillo con edema y dificultad para la deambulación",
        "gonalgia y derrame articular tras traumatismo deportivo",
    ],
}
prioridad_por_esp = {
    "cardiología": [0.35, 0.45, 0.20],
    "respiratorio": [0.30, 0.45, 0.25],
    "digestivo": [0.20, 0.50, 0.30],
    "neurología": [0.45, 0.35, 0.20],
    "traumatología": [0.12, 0.48, 0.40],
}
tiempos = ["horas", "un día", "dos días", "una semana", "varios días"]
n_notas = 3000
esp_list = list(plantillas.keys())
notas_rows = []
for _ in range(n_notas):
    esp = RNG.choice(esp_list)
    txt = RNG.choice(plantillas[esp]).replace("{t}", RNG.choice(tiempos))
    prio = RNG.choice(["alta", "media", "baja"], p=prioridad_por_esp[esp])
    centro = RNG.choice(centros["centro_id"].values)
    notas_rows.append((txt, esp, prio, centro))
notas = pd.DataFrame(notas_rows, columns=["texto", "especialidad", "prioridad", "centro_id"])
notas.to_csv(OUT / "notas_clinicas.csv", index=False)

# =====================================================================
# 6) WEARABLE (opcional)  (wearable.csv)  — señal sencilla para U7/U8
#    200 sujetos × 30 días: frecuencia cardiaca en reposo, pasos, horas de sueño.
#    Se inyectan algunas anomalías para detección.
# =====================================================================
n_suj, n_dias_w = 200, 30
rows_w = []
for s in range(1, n_suj + 1):
    fc_base = RNG.normal(66, 6)
    pasos_base = RNG.normal(7500, 2500)
    sueno_base = RNG.normal(7.0, 0.8)
    anomalo = RNG.random() < 0.05     # ~5% sujetos con días anómalos
    for dia in range(1, n_dias_w + 1):
        fc = fc_base + RNG.normal(0, 3)
        pasos = max(0, pasos_base + RNG.normal(0, 1500) - (1500 if dia % 7 in (0, 6) else 0) * 0)
        sueno = np.clip(sueno_base + RNG.normal(0, 0.6), 3, 11)
        if anomalo and dia in (14, 15, 16):
            fc += RNG.uniform(18, 30)     # episodio: taquicardia sostenida
            sueno -= RNG.uniform(1.5, 3)
        rows_w.append((f"S{s:03d}", dia, round(fc, 1), int(pasos), round(float(sueno), 1)))
wearable = pd.DataFrame(rows_w, columns=["sujeto_id", "dia", "fc_reposo", "pasos", "horas_sueno"])
wearable.to_csv(OUT / "wearable.csv", index=False)

# =====================================================================
# RESUMEN / VERIFICACIÓN
# =====================================================================
print("=== pacientes.csv ===", pacientes.shape)
print("prevalencia evento_cv:", round(pacientes.evento_cv.mean(), 3),
      "| prevalencia diabetes:", round(pacientes.diabetes.mean(), 3))
print("riesgo_cv_10a media/min/max:", round(pacientes.riesgo_cv_10a.mean(), 1),
      pacientes.riesgo_cv_10a.min(), pacientes.riesgo_cv_10a.max())
print("edad media:", round(pacientes.edad.mean(), 1), "| IMC medio:", round(pacientes.imc.mean(), 1))
print("evento_cv por sexo:\n", pacientes.groupby("sexo").evento_cv.mean().round(3))
print("evento_cv por tabaquismo:\n", pacientes.groupby("tabaquismo").evento_cv.mean().round(3))
corr = pacientes[["edad", "imc", "ta_sistolica", "glucemia", "hdl", "riesgo_cv_10a"]].corr()["riesgo_cv_10a"].round(2)
print("corr con riesgo_cv_10a:\n", corr)
print("\n=== urgencias_diarias.csv ===", urgencias.shape,
      "| ingresos medios:", round(urgencias.ingresos.mean(), 1))
print("media invierno vs verano:",
      round(urgencias[pd.to_datetime(urgencias.fecha).dt.month.isin([12,1,2])].ingresos.mean(),1), "vs",
      round(urgencias[pd.to_datetime(urgencias.fecha).dt.month.isin([6,7,8])].ingresos.mean(),1))
print("\n=== notas_clinicas.csv ===", notas.shape, "| especialidades:", notas.especialidad.nunique())
print("\n=== centros.csv ===", centros.shape)
print("\n=== wearable.csv ===", wearable.shape)
print("\n=== pacientes_sucio.csv ===", d.shape, "(con duplicados y suciedad)")
print("\nOK: todos los CSV generados en", OUT.resolve())
