---
description: >-
  Guía de referencia completa para dirigir a un agente de IA a lo largo de un
  proyecto clínico de principio a fin: contratos previos, contexto, el bucle
  de evidencia, prompts reutilizables y el papel del profesional sanitario al
  mando. Documento de estudio complementario. Todos los datos son sintéticos.
---

# Copiloto de ciencia de datos (guía end-to-end)

{% hint style="info" %}
**Material complementario del curso · guía de referencia**

Este documento **une todo lo visto en el curso** con la nueva forma de trabajar que presenta la [**Unidad 10**](../unidades/u10-copiloto-datascience.md). La U10 te enseñó la idea y te la hizo sentir con un ejemplo guiado. Esta guía es la **pieza de consulta**: la que abres cuando vas a dirigir tú mismo un análisis real, de principio a fin, y quieres tener a mano los contratos previos, la plantilla de contexto, las fases del proceso y los prompts que puedes copiar y adaptar.

Es hermana del estudio [**Bucles de auto-mejora (prompting-loops.md)**](prompting-loops.md), que profundiza en cómo un agente itera y ajusta su propia búsqueda; esta guía mira el **flujo completo**, del encargo al informe final.

{% endhint %}

{% hint style="warning" %}
**⚠️ Aviso · Datos sintéticos**

Todos los ejemplos de esta guía usan la cohorte sintética del curso ([`pacientes.csv`](https://drive.google.com/file/d/1Ku0j-sAf8Cr3FPT-DGm8v5p4h_2BmV5U/view?usp=drive_link) y ficheros hermanos: [`pacientes_sucio.csv`](https://drive.google.com/file/d/1geOuVmKqhBvBf52NyVFBvpxw7Ypbucp2/view?usp=drive_link), [`urgencias_diarias.csv`](https://drive.google.com/file/d/1EpQ9Lcb-f-iDqBOA3f3sT_pGLBp2G56u/view?usp=drive_link), [`notas_clinicas.csv`](https://drive.google.com/file/d/1cWvZFsNd1d-Wd_B8G2eTLewqyjiydE0x/view?usp=drive_link), [`centros.csv`](https://drive.google.com/file/d/1rxxkSTg-hsyiLlC6ppGpKAoMjmBrolM-/view?usp=drive_link), [`wearable.csv`](https://drive.google.com/file/d/1az7oq8Rzkts0u37ijWVaRTvUnmpbNU7o/view?usp=drive_link)). Ningún dato representa pacientes reales. Su única función es didáctica: aprender a dirigir un proyecto de principio a fin sin necesidad de escribir código.
{% endhint %}

La forma de trabajar en ciencia de datos ha cambiado. No porque los fundamentos hayan desaparecido —al contrario, **importan más que antes**—, sino porque ha cambiado la unidad de trabajo.

Antes, un proyecto de análisis clínico era algo así:

```text
explorar datos → limpiar → construir variables → probar modelos → ajustar → evaluar → informe
```

En 2026, cada vez más, un proyecto se parece a esto:

```text
objetivo clínico + contexto + datos + restricciones
        ↓
bucle de experimentación dirigido por ti, ejecutado por un agente
        ↓
evidencia, modelos, gráficos, errores, decisiones
        ↓
modelo final + informe reproducible + criterio de cuándo NO usarlo
```

La diferencia parece pequeña, pero no lo es. Es como pasar de cocinar siguiendo una receta a dirigir una cocina con varios ayudantes que cortan, prueban, comparan y te van diciendo qué está fallando. El profesional sigue siendo responsable del plato. Pero ya no corta cada cebolla.

La idea central de esta guía es la misma que abrió la U10:

> **Ya no programas el modelo. Diriges un bucle de evidencia.**

Y en ese bucle entran modelos estadísticos clásicos, tablas, gráficos, código ejecutado, interpretación clínica y, sobre todo, tu criterio.

No es "la IA reemplaza al profesional sanitario que analiza datos". Esa frase es pobre. La versión buena es esta:

> **La IA convierte muchas tareas de análisis en procesos iterativos y verificables, siempre que el humano sepa formular el objetivo, aportar contexto clínico y evaluar la evidencia.**

La revolución no consiste en apretar un botón. Consiste en aprender a **dirigir bien**.

---

## Índice

1. [Los seis contratos de un proyecto clínico con IA](#1-los-seis-contratos-de-un-proyecto-clínico-con-ia)
2. [Qué contexto debe recibir el agente](#2-qué-contexto-debe-recibir-el-agente)
3. [Plantilla de contexto para entregar al agente](#3-plantilla-de-contexto-para-entregar-al-agente)
4. [El proceso completo, fase a fase](#4-el-proceso-completo-fase-a-fase)
5. [El bucle de evidencia explícito](#5-el-bucle-de-evidencia-explícito)
6. [Modo conversación y modo encargo amplio](#6-modo-conversación-y-modo-encargo-amplio)
7. [Prompts reutilizables](#7-prompts-reutilizables)
8. [Lo que el agente no debe decidir solo](#8-lo-que-el-agente-no-debe-decidir-solo)
9. [Antipatrones frecuentes](#9-antipatrones-frecuentes)
10. [Dos encargos completos de ejemplo](#10-dos-encargos-completos-de-ejemplo)
11. [Cómo pedir evidencia, no opiniones](#11-cómo-pedir-evidencia-no-opiniones)
12. [Qué llevarte](#qué-llevarte)

---

## 1. Los seis contratos de un proyecto clínico con IA

Antes de pedirle nada a un agente, un proyecto serio debería quedar definido mediante **contratos**. No contratos legales: acuerdos explícitos contigo mismo (y con el agente) sobre qué se va a hacer y con qué reglas. Es la misma disciplina que ya aplicaste, sin llamarla así, en la U3 al fijar la métrica y la validación.

### 1.1 Contrato de problema

Define qué se quiere predecir o entender, y por qué.

```text
¿Qué decisión clínica se tomará con el resultado?
¿Quién la usará: el propio clínico, el equipo de gestión, un sistema de alertas?
¿Qué pasa si el modelo se equivoca?
¿Qué tipo de error es más caro: no detectar un caso, o alertar de más?
¿La predicción se hace una vez, cada consulta, cada día?
¿Es clasificación (evento sí/no), regresión (un valor continuo), o una serie temporal?
```

Ejemplo malo:

```text
Predecir riesgo cardiovascular.
```

Ejemplo bueno:

```text
Predecir si un paciente sufrirá un evento cardiovascular en los próximos 10 años,
usando solo la información disponible en la consulta de hoy. El resultado se usará
para priorizar a quién citar antes en el programa de prevención. Nos importa mucho
más NO dejar fuera a un paciente que sí tendrá el evento (falso negativo) que citar
de más a uno de bajo riesgo, pero tampoco queremos saturar la agenda de prevención
con falsos positivos.
```

Ese texto vale oro. Es, casi palabra por palabra, el contrato de problema que la U10 trabajó sobre `evento_cv`.

### 1.2 Contrato de datos

Define qué representa cada fila.

```text
Una fila, ¿es un paciente, una visita, un día, un centro?
¿Cuál es la clave que identifica cada fila (p. ej. paciente_id)?
¿Puede haber varias filas por paciente?
¿Qué columnas son fechas?
¿Qué columnas son identificadores (centro_id, sujeto_id)?
¿Qué columnas son texto libre (notas_clinicas.csv)?
¿Qué columnas podrían existir solo DESPUÉS del evento que quieres predecir?
```

La pregunta "qué representa una fila" parece trivial. No lo es. En `pacientes.csv`, una fila es un paciente; en `urgencias_diarias.csv`, un día; en `wearable.csv`, un sujeto-día. Confundir la unidad de análisis es el origen de muchos errores de validación (partir por fila cuando hay varias filas por paciente, por ejemplo).

### 1.3 Contrato de objetivo (target)

Define la variable que se quiere predecir.

```text
¿Cómo se construye exactamente?
¿En qué ventana temporal se observa?
¿Hay ruido en la etiqueta (un evento mal registrado, una fecha aproximada)?
¿La etiqueta se observa directamente o se infiere de otras columnas?
¿Hay casos ambiguos?
¿Hay retraso entre que ocurre el evento y que queda registrado?
```

Ejemplo, aplicado al hilo del curso:

```text
target = evento_cv = 1 si el paciente sufre un evento cardiovascular
         dentro del horizonte de seguimiento del estudio.
target = 0 si no consta evento en ese periodo.
Se excluyen pacientes con seguimiento incompleto.
```

Esto evita uno de los grandes pecados: entrenar con una etiqueta mal definida sin haberlo advertido.

### 1.4 Contrato de métrica

La métrica es la brújula. Si la brújula está mal, el agente corre —con toda su diligencia— en la dirección clínica equivocada. Esto ya se trabajó a fondo en la U3; aquí solo se recuerda como contrato explícito:

```text
Clasificación con clase minoritaria (evento_cv, prevalencia ≈19%):
sensibilidad/recall, PR-AUC, sensibilidad en el grupo priorizado.

Riesgo con coste asimétrico de errores:
recall bajo restricción de falsos positivos, coste esperado por error.

Regresión (riesgo_cv_10a):
MAE, RMSE, R².

Series temporales (urgencias_diarias.csv):
MAE, RMSE, error por temporada.
```

La métrica debe conectar con la decisión clínica:

```text
Una accuracy alta puede ser inútil: con prevalencia ≈19%, decir siempre "no"
acierta el 81% de las veces y no detecta a nadie.
Un AUC "demasiado bueno" (p. ej. > 0,97) es motivo de sospecha, no de celebración:
suele ser síntoma de fuga de datos, no de un modelo excelente.
```

Una métrica mal elegida es una forma elegante de equivocarse con seguridad.

### 1.5 Contrato de validación

Define cómo se comprueba que el modelo generaliza y no ha memorizado.

```text
partición aleatoria por fila (solo si cada fila es una entidad distinta)
partición por paciente (si hay varias filas por persona)
partición temporal (si hay una fecha de corte, como en urgencias_diarias.csv)
validación cruzada (k-fold) para comparar modelos con más confianza
holdout final, congelado, que solo se mira una vez
```

Aquí hay que ser muy estricto. La regla simple, que ya conoces de la U3 y de la U10:

```text
Todo lo que aprenda algo de los datos (imputar, escalar, codificar) debe
aprenderlo SOLO dentro del conjunto de entrenamiento, nunca mirando el test.
```

Eso incluye imputación de valores ausentes, escalado, codificación de categorías y cualquier selección de variables.

### 1.6 Contrato de uso clínico

Define las condiciones reales de uso, sin entrar en jerga de infraestructura. La pregunta que importa a un no programador no es "¿en qué servidor corre?", sino:

```text
¿El resultado debe poder explicarse a un paciente o a un comité clínico?
¿Se usa en el momento de la consulta o en un análisis retrospectivo?
¿Con qué frecuencia habría que revisar si el modelo sigue funcionando bien?
¿Quién es la persona responsable de decidir si se sigue usando o se retira?
¿Qué pasa si el modelo "se equivoca en silencio" (nadie se da cuenta)?
```

Este contrato puede cambiar el modelo preferido. Un modelo algo más complejo puede ganar unas décimas de AUC, pero ser inviable porque nadie en el comité clínico puede explicar por qué decide lo que decide. Un modelo más simple —como la regresión logística que gana en `evento_cv`— puede perder un poco de métrica y ganar muchísimo en confianza clínica y en facilidad para vigilar que sigue funcionando bien.

---

## 2. Qué contexto debe recibir el agente

El dataset no basta. El contexto es el combustible real, y es exactamente lo que trabajó la sección 10.2 de la U10 con el ejemplo del prompt pobre frente al prompt con contexto clínico.

Un buen encargo inicial debe incluir, como mínimo:

```text
1. Descripción del problema clínico.
2. Qué representa cada fila.
3. Variable objetivo y cómo se define.
4. Momento en el que se toma la decisión (qué información existe entonces).
5. Métrica principal y, si aplica, una secundaria.
6. Qué tipo de error es más caro (falso positivo o falso negativo).
7. Si el modelo necesita ser explicable a un comité clínico.
8. Riesgos conocidos de fuga de datos (columnas "del futuro").
9. Grupos o segmentos clínicos importantes de evaluar por separado.
10. Información temporal relevante.
11. Casos difíciles o ambiguos que ya conozcas.
12. Qué significa "éxito" para este análisis.
13. Qué significa "no usar este modelo".
```

La mayoría de encargos pobres fallan porque dicen:

```text
Aquí tienes pacientes.csv, hazme el mejor modelo.
```

Los encargos buenos dicen:

```text
Analiza pacientes.csv para resolver este problema clínico concreto, con estas
restricciones, esta métrica, esta definición de la ventana temporal y estos
riesgos de que algo esté mal medido.
```

El agente no lee mentes clínicas. Pero con buen contexto puede exprimir su conocimiento estadístico de forma impresionante.

---

## 3. Plantilla de contexto para entregar al agente

Esta plantilla es la versión de referencia, más extensa que la de la U10, pensada para copiar y rellenar antes de cualquier análisis nuevo con datos clínicos.

```markdown
# Encargo de análisis clínico

## 1. Objetivo
Queremos predecir / entender ...

## 2. Decisión clínica que apoyará el resultado
El resultado se usará para ...

## 3. Unidad de análisis
Cada fila representa ... (paciente / paciente-visita / centro-día / sujeto-día)

## 4. Objetivo a predecir (target)
La variable objetivo es ...
Se define como ...
Se observa en ... (ventana temporal, momento del seguimiento)

## 5. Momento de la decisión
El análisis se usaría en ...
Solo puede usar información disponible hasta ese momento.

## 6. Datos
Archivo principal: ... (p. ej. pacientes.csv — SINTÉTICO)
Número aproximado de filas: ...
Columnas conocidas:
- ...
- ...

## 7. Métrica
Métrica principal: ...
Métrica secundaria (si aplica): ...
Restricciones mínimas:
- sensibilidad >= ...
- especificidad/precisión >= ...

## 8. Coste de los errores
Falso positivo (alertar de más): ...
Falso negativo (no detectar un caso real): ...

## 9. Validación
Usar:
- partición por paciente / partición temporal / validación cruzada
No usar:
- partición aleatoria por fila si hay varias filas por paciente
- variables que solo existen después del evento

## 10. Restricciones de modelo
Preferimos:
- un modelo que se pueda explicar a un comité clínico / la mejor métrica posible
No permitimos:
- variables no disponibles en el momento de decidir
- resultados que no se puedan reproducir y revisar

## 11. Riesgos conocidos
- fuga de datos por ...
- etiqueta con ruido por ...
- valores ausentes en ...
- clase minoritaria: prevalencia aproximada ...

## 12. Grupos clínicos importantes
Evaluar el rendimiento por separado en:
- edad / sexo
- tabaquismo (nunca / ex / activo)
- centro o área
- presencia de diabetes o antecedentes familiares

## 13. Qué debe entregarme
- resumen de la exploración inicial de los datos
- informe de calidad de datos
- análisis de posibles fugas de datos
- comparación entre varios modelos candidatos (con un baseline simple primero)
- evaluación final, incluida por grupos clínicos
- gráficos que apoyen la interpretación
- análisis de los errores más relevantes
- una explicación en lenguaje llano de la recomendación final
```

Esto convierte un encargo ambiguo en un proyecto que el agente puede ejecutar con criterio.

---

## 4. El proceso completo, fase a fase

Este recorrido amplía, fase por fase, el bucle que la U10 resumió en cinco pasos (proponer, ejecutar, medir, interpretar, validar). Aquí lo desdoblamos para que sirva de checklist cuando el análisis es más largo o más delicado.

### Fase 0 — Formular el problema antes de tocar el CSV

La primera tarea del agente no debería ser analizar. Debería ser preguntar.

Prompt:

```text
Actúa como mi copiloto de ciencia de datos clínicos. Antes de tocar los datos,
revisa la descripción de mi problema y dime qué ambigüedades críticas ves.
Clasifícalas en:
1. ambigüedades que bloquean cualquier análisis serio,
2. ambigüedades que afectan a qué métrica debería usar,
3. ambigüedades que afectan al riesgo de fuga de datos,
4. ambigüedades que puedes resolver tú con un supuesto razonable (dímelo).

Luego propón una versión formal del problema.
```

El resultado esperado no es código. Es claridad, como en el ejemplo de la sección 10.1 de la U10.

### Fase 1 — Auditoría inicial de los datos

Aquí empieza la exploración, pero no como "hacer gráficos bonitos" (esto ya lo trabajaste a fondo en la U2). Es descubrir la estructura, los riesgos y las oportunidades del conjunto de datos.

Prompt:

```text
Haz una auditoría inicial de pacientes.csv. No entrenes ningún modelo todavía.

Quiero:
1. tipos de cada columna y qué representa clínicamente,
2. porcentaje de valores ausentes por columna,
3. columnas que parecen identificadores,
4. distribución de evento_cv (¿cuál es la prevalencia real?),
5. valores imposibles o rangos sospechosos (p. ej. una TA sistólica de 300),
6. columnas que podrían producir fuga de datos,
7. un resumen de riesgos antes de seguir.
```

Aquí un agente puede sorprender porque no se limita a un resumen estadístico plano: puede fijarse en los nombres de las columnas, en relaciones entre variables (¿la glucemia es coherente con el diagnóstico de diabetes?) y en consistencias clínicas.

### Fase 2 — Calidad de datos como comprobaciones, no como intuición

Un error frecuente es tratar la limpieza como algo artesanal ("quitamos los raros y ya"). Conviene, en cambio, pensarla como un conjunto de **comprobaciones explícitas**, tal como se practicó sobre `pacientes_sucio.csv` en la U2.

Ejemplos de comprobaciones clínicas:

```text
La edad debe estar entre 18 y 95 (según el diseño de la cohorte).
La TA sistólica debe ser mayor que la TA diastólica.
La glucemia no puede ser negativa.
El HDL no puede ser mayor que el colesterol total.
Cada paciente_id debe aparecer una sola vez.
El tabaquismo debe ser uno de: nunca, ex, activo.
```

Prompt:

```text
Diseña un conjunto de comprobaciones de calidad para pacientes.csv.

Divide las comprobaciones en:
1. rangos clínicamente válidos,
2. consistencia entre columnas (p. ej. TA sistólica vs. diastólica),
3. unicidad de paciente_id,
4. categorías permitidas,
5. comprobaciones que necesitan mi confirmación clínica.

Para cada una, dime: razón, gravedad si falla, y qué hacer si falla.
```

La clave didáctica, la misma que ya viste en la U2:

> **No limpiamos datos para que el modelo funcione. Limpiamos datos para que el análisis sea verdadero.**

### Fase 3 — Valores ausentes: no imputar antes de entender

Los valores ausentes no son solo una molestia técnica. A veces son señal clínica.

```text
MCAR: falta completamente al azar (p. ej. un fallo de registro sin relación con nada).
MAR: falta según otras variables observadas (p. ej. falta más en pacientes de un centro concreto).
MNAR: falta por una razón relacionada con el propio valor (p. ej. no se registra
      el HDL precisamente en los pacientes que no se hicieron la analítica completa).
```

Prompt:

```text
Analiza los valores ausentes en pacientes.csv antes de imputar nada.

Quiero:
1. qué columnas tienen más ausencias,
2. si la ausencia está relacionada con evento_cv,
3. una hipótesis sobre por qué falta cada tipo de dato,
4. si conviene crear una columna "dato_ausente" además de imputar,
5. una comparación entre imputación simple e imputación con indicador.

Toda imputación debe hacerse dentro del esquema de validación, nunca antes.
```

Una regla para no programadores:

```text
No imputes porque el modelo se queja de que faltan datos.
Imputa porque entiendes qué significa clínicamente que falte ese dato.
```

### Fase 4 — Valores atípicos: clasificar, no borrar

En clínica, un valor atípico casi nunca es "ruido a eliminar". Puede ser el caso más importante del dataset.

```text
glucemia = 380
```

puede ser un paciente diabético mal controlado: no es un error, es información clínica valiosa.

Pero:

```text
edad = 12
diabetes = 1
tabaquismo = activo
```

sí merece revisión: es una combinación clínicamente improbable en la cohorte descrita.

Prompt:

```text
Detecta valores atípicos en pacientes.csv en varias capas:
1. valores imposibles por definición (p. ej. edad fuera de 18-95),
2. valores estadísticamente extremos pero univariantes,
3. combinaciones de variables clínicamente inconsistentes,
4. atípicos por subgrupo (p. ej. raro para su rango de edad).

No elimines nada automáticamente. Clasifica cada caso como:
- error de registro probable,
- valor raro pero clínicamente plausible,
- caso potencialmente importante (no descartar),
- necesita revisión clínica.

Dame ejemplos concretos y tu recomendación para cada grupo.
```

### Fase 5 — Análisis de fuga de datos: la fase de mayor valor clínico

La fuga de datos es el gran enemigo de cualquier análisis, y también el riesgo que la U10 señaló como el más peligroso de todos porque **no chirría**: un resultado "demasiado bueno" se acepta con gusto, y ahí está el problema.

Tipos de fuga habituales en el mundo clínico:

```text
1. Variables que solo existen después de que ocurra el evento
   (p. ej. una fecha de ingreso por el propio evento cardiovascular).
2. Variables agregadas que usan información del futuro respecto al momento de decidir.
3. Duplicados del mismo paciente repartidos entre entrenamiento y test.
4. Varias filas del mismo paciente separadas al azar en vez de agrupadas.
5. Identificadores de centro con una correlación artificial con el desenlace.
6. Escalado o imputación calculados fuera del conjunto de entrenamiento.
```

Prompt:

```text
Haz un análisis exigente de fuga de datos sobre pacientes.csv.

Para cada columna, clasifícala como:
- segura,
- sospechosa,
- probablemente fuga,
- necesita mi conocimiento clínico para decidir.

Busca en especial:
1. columnas que solo tendrían sentido después del evento,
2. correlaciones sospechosamente altas con evento_cv,
3. identificadores que podrían "memorizar" en vez de generalizar,
4. transformaciones que deberían calcularse solo dentro del entrenamiento.

No entrenes el modelo final hasta entregarme este informe de fuga de datos.
```

El informe de fuga debería tener esta forma:

```text
columna | riesgo | evidencia | decisión | comentario clínico
```

Ejemplo:

```text
fecha_evento_cv | fuga probable | solo existe si ya hubo evento | excluir | no disponible al decidir
centro_id | riesgo de memorización | alta cardinalidad, pocos pacientes por centro | usar con cautela | depende de la partición
```

Para un análisis clínico serio, esta fase vale más que cualquier comparación adicional de modelos.

### Fase 6 — Baselines: el antídoto contra la fantasía

Antes de probar modelos más sofisticados, hay que fijar un punto de referencia sencillo, como ya se hizo en la U4 y en la U10.

```text
baseline ingenuo (p. ej. predecir siempre la clase mayoritaria)
baseline interpretable (regresión logística o regresión lineal)
modelo más flexible (Random Forest, boosting)
```

Prompt:

```text
Construye baselines antes de probar modelos más complejos sobre pacientes.csv.

Incluye:
1. un baseline trivial (predecir siempre "no evento"),
2. una regresión logística como baseline interpretable,
3. comparación con la métrica principal y, si aplica, la secundaria,
4. una explicación de por qué cada baseline importa como referencia.

No hagas todavía ajuste fino de hiperparámetros.
```

Un baseline es una piedra en el suelo. Evita que el análisis flote en cifras sin referencia, y es lo que permite decir con propiedad que "la logística gana a Random Forest" en `evento_cv`, en vez de limitarse a reportar un número aislado.

### Fase 7 — Modelos candidatos

En 2026, elegir modelos no debería ser dogmático. Depende del problema, tal como viste en las unidades U4 a U8.

Para datos tabulares como `pacientes.csv` conviene probar, por este orden:

```text
regresión logística / regresión lineal (según el objetivo)
Random Forest
gradient boosting
```

Para texto libre como `notas_clinicas.csv`:

```text
representación simple del texto + modelo lineal
modelos de lenguaje ya entrenados (U9), si el volumen y el caso lo justifican
```

Para series temporales como `urgencias_diarias.csv`:

```text
un valor de referencia simple (el mismo día de la semana pasada)
modelos con variables de calendario y rezagos (U7)
```

Prompt:

```text
Propón una lista de modelos candidatos razonables para este problema.

Para cada modelo, dime:
1. por qué tiene sentido para este dataset y este objetivo clínico,
2. qué preparación de datos necesita,
3. si es fácil de explicar a un comité clínico,
4. qué riesgo de sobreajuste tiene,
5. cómo lo vas a evaluar de forma justa (misma validación para todos).

Luego ejecuta una primera ronda con presupuesto de tiempo limitado.
```

### Fase 8 — Ajuste de hiperparámetros

El ajuste fino no debería hacerse a ciegas ni de forma indefinida.

Prompt:

```text
Ajusta los hiperparámetros del modelo más prometedor.

Reglas:
1. Usa solo entrenamiento y validación, nunca el test final.
2. Respeta el esquema de partición acordado (por paciente).
3. Compara siempre contra el baseline sin ajustar.
4. Guarda cada intento (qué probaste y qué resultado dio).
5. Dime cuándo conviene parar.

Al terminar, entrégame: la mejor configuración, cómo varía la métrica entre
particiones, y la comparación final con el baseline.
```

Un buen agente no debería ajustar de forma infinita. Debería saber parar cuando:

```text
la mejora ya no es relevante clínicamente tras varios intentos,
el modelo más complejo no supera claramente al baseline simple,
la métrica varía mucho entre particiones (señal de inestabilidad).
```

### Fase 9 — Evaluación: más allá de un único número

El error clásico:

```text
Modelo A: AUC = 0,842
Modelo B: AUC = 0,846
Gana B.
```

Eso es demasiado pobre para un contexto clínico. Una evaluación seria, como ya trabajaste en la U3, debe mirar:

```text
métrica principal y secundaria
variación entre particiones
rendimiento por grupo clínico (edad, tabaquismo, centro)
matriz de confusión
calibración (¿un 30% de riesgo predicho significa realmente un 30% de eventos?)
ajuste del umbral de decisión
casos donde el modelo se equivoca con más confianza
```

Prompt:

```text
Haz una evaluación completa del mejor modelo y de los dos siguientes candidatos.

Incluye:
1. tabla de métricas globales,
2. variación entre particiones de validación cruzada,
3. matriz de confusión,
4. curva de calibración,
5. rendimiento por grupo (tabaquismo, edad, centro),
6. los errores más costosos y qué tienen en común,
7. comparación con el baseline,
8. tu recomendación final, justificada.

No toques el holdout final hasta que el análisis esté congelado.
```

### Fase 10 — Ajuste del umbral de decisión

En clasificación, muchas veces el modelo no está mal: lo que está mal es el punto de corte que separa "alerta" de "no alerta".

```text
umbral = 0,5
```

es una convención, no una ley clínica. Si lo que importa es no dejar fuera a nadie con riesgo real, puede convenir un umbral más bajo, aunque eso genere más falsos positivos.

Prompt:

```text
Ajusta el umbral de decisión usando solo el conjunto de validación.

Genera:
1. cómo cambian sensibilidad y precisión según el umbral,
2. una recomendación de umbral según la restricción clínica:
   - sensibilidad mínima aceptable, o
   - capacidad real del equipo de prevención para citar pacientes.

No ajustes el umbral mirando el test final.
```

### Fase 11 — Interpretación clínica

La interpretación no es solo "sacar la importancia de las variables". Hay varias capas, como ya se vio en la U5 con SHAP:

```text
qué variables importan en general (interpretación global)
por qué esta predicción concreta, para este paciente (interpretación local)
qué patrón aparece en los falsos positivos y en los falsos negativos
```

Prompt:

```text
Genera un análisis de interpretación del modelo elegido.

Incluye:
1. qué variables pesan más en general,
2. explicación de 5 predicciones correctas representativas,
3. explicación de 5 falsos negativos (los más preocupantes clínicamente),
4. si los efectos observados (tabaquismo, HDL, actividad física) tienen
   sentido clínico,
5. un resumen en lenguaje llano para presentar a un comité no técnico.
```

Una buena pregunta para el agente, la misma que planteó la U10:

```text
¿Qué explicación del modelo cambiaría una decisión clínica?
```

Si la explicación no cambiaría nada, quizá es solo decoración.

### Fase 12 — Análisis de errores: donde realmente se aprende

No basta con decir "el modelo falla un 15% de las veces". Hay que preguntar:

```text
¿Falla más en un grupo de edad concreto?
¿Falla más en pacientes con un dato ausente?
¿Falla por una etiqueta mal registrada?
¿Falla en casos clínicamente ambiguos de verdad?
```

Prompt:

```text
Haz un análisis de errores del modelo actual.

Divide los errores en:
1. falsos positivos (alertamos de más),
2. falsos negativos (no detectamos un caso real),
3. errores con alta confianza del modelo (los más preocupantes),
4. errores por grupo clínico,
5. errores asociados a valores ausentes.

Para cada grupo, propón una explicación posible y qué harías para investigarla.
```

Este es el bucle más importante de todos:

```text
error → hipótesis clínica → variable o dato revisado → comprobación → conclusión
```

### Fase 13 — Rendimiento por grupos clínicos

En salud, la métrica global casi nunca basta. Un modelo puede tener una sensibilidad global excelente y fallar sistemáticamente en un subgrupo (por ejemplo, mujeres, o pacientes de un centro concreto).

Prompt:

```text
Evalúa el rendimiento del modelo por separado en estos grupos:
- sexo,
- rango de edad,
- tabaquismo (nunca / ex / activo),
- presencia de diabetes,
- centro (si aplica).

Para cada grupo, dame el tamaño de muestra y la métrica principal.
Señala qué grupos tienen un rendimiento claramente peor y qué podría explicarlo.
```

No todo análisis requiere un estudio formal de sesgo. Pero casi todo análisis clínico serio requiere esta **evaluación por grupos**, aunque solo sea para descubrir a tiempo que el modelo "funciona en general" pero falla justo en el grupo que más te importa.

### Fase 14 — Informe final

La documentación no es burocracia. Es la memoria del análisis, y sin ella un resultado no se puede defender ni reproducir dentro de seis meses.

Prompt:

```text
Genera un informe final del análisis en lenguaje comprensible para un
profesional sanitario, no necesariamente técnico.

Estructura:
1. Resumen en dos párrafos.
2. El problema clínico y la decisión que apoya.
3. Los datos usados (recuerda que son SINTÉTICOS) y qué representa cada fila.
4. Cómo se validó (sin fuga de datos).
5. Qué se descubrió en la exploración inicial.
6. Qué modelos se compararon y cuál ganó, y por qué.
7. Cómo se comporta el modelo por grupo clínico.
8. Los errores más relevantes y qué significan.
9. Limitaciones honestas del análisis.
10. Recomendación final: ¿se podría usar? ¿bajo qué condiciones? ¿qué se
    debería revisar antes?
```

Para quien dirige el análisis, esta fase enseña algo crucial: **un resultado sin informe es una predicción sin memoria**.

---

## 5. El bucle de evidencia explícito

Todo lo anterior se puede resumir en el ciclo que ya presentó la U10, aquí con más detalle para cuando el proyecto es más largo.

```text
mientras quede algo razonable que probar:

    leer el estado actual del análisis
    revisar qué ha funcionado y qué no hasta ahora
    analizar los errores del mejor modelo actual
    proponer una hipótesis clínica concreta
    priorizar qué hipótesis probar primero
    ejecutar el experimento (código real, no una promesa)
    medir con la métrica acordada
    comparar contra el mejor resultado anterior
    interpretar clínicamente el resultado
    decidir: conservar, descartar, investigar más, o parar
```

Versión más formal, útil para pedírsela directamente a un agente al inicio de un proyecto largo:

```text
Estado que debes mantener:
- los datos y su contexto clínico
- las reglas de validación acordadas
- la métrica acordada
- qué experimentos ya se han hecho y con qué resultado
- qué errores conocidos hay
- qué restricciones no se pueden romper

En cada vuelta:
- Propón una hipótesis clínica concreta.
- Ejecuta el experimento correspondiente.
- Mídelo con la métrica acordada.
- Interpreta el resultado en términos clínicos.
- Dime si lo conservas, lo descartas o necesitas mi decisión.

Criterios para parar:
- se alcanza el objetivo clínico fijado,
- varios experimentos seguidos no mejoran de forma relevante,
- el modelo más simple sigue siendo la mejor opción,
- se detecta un riesgo (de fuga de datos, de sesgo) que hay que resolver antes de seguir.
```

Este es exactamente el patrón que la U10 llamó *loop engineering*, aplicado ahora a un proyecto completo en vez de a una sola comparación de modelos.

---

## 6. Modo conversación y modo encargo amplio

Hay dos formas principales de dirigir a un agente, y conviene saber cuándo usar cada una.

### 6.1 Modo conversación (centauro interactivo)

Es el modo chat: tú sigues controlando cada paso, el agente propone, ejecuta trozos de código, interpreta resultados y te pregunta cuando algo no está claro.

Es ideal para:

```text
aprender mientras se analiza
explorar un dataset nuevo por primera vez
decisiones clínicas que hay que tomar con frecuencia durante el análisis
cuando el problema todavía no está bien definido
```

La dinámica típica es justo la que viste en la sección 10.3 de la U10: el agente propone, tú aportas el contexto que le falta, y el análisis avanza vuelta a vuelta.

### 6.2 Modo encargo amplio (casi autónomo)

Aquí defines de una vez un encargo extenso, con contrato de problema, contexto y criterios de parada, y dejas que el agente recorra varias fases seguidas antes de volver a interrumpirte, deteniéndose solo en las decisiones que tú marcaste como importantes.

Es ideal para:

```text
proyectos con el problema ya bien formalizado
repetir un mismo tipo de análisis sobre datos nuevos
cuando quieres un informe completo y no necesitas revisar cada paso intermedio
```

La regla práctica:

```text
Si el problema clínico todavía es ambiguo, empieza en modo conversación.
Si el problema ya está formalizado (con sus seis contratos), pasa a un encargo amplio.
```

| Aspecto | Modo conversación | Modo encargo amplio |
|---|---|---|
| Control | Tú decides cada paso importante | El agente recorre el bucle y se detiene en los puntos que fijaste |
| Ideal para | Aprender, explorar, decidir con frecuencia | Repetir, automatizar, producir un informe completo |
| Riesgo principal | Perder de vista el conjunto si hay muchas idas y vueltas | Que el objetivo esté mal formulado y el agente lo optimice con diligencia |
| Necesidad de contexto | Alta | Altísima, porque hay menos ocasiones de corregir sobre la marcha |

---

## 7. Prompts reutilizables

Esta sección reúne, en un solo sitio, los prompts de cada fase para copiar y adaptar. Son deliberadamente parecidos entre sí: la disciplina importa más que la originalidad.

### 7.1 Prompt maestro de arranque

```text
Quiero que actúes como mi copiloto de ciencia de datos clínicos.

Voy a darte un dataset (SINTÉTICO) y el contexto de un problema clínico.
Tu trabajo no es solo entrenar un modelo: es ayudarme a construir el mejor
análisis posible, de forma rigurosa y explicándome cada decisión.

Sigue este orden:
1. formaliza el problema y señala ambigüedades,
2. audita los datos,
3. analiza valores ausentes, valores atípicos y riesgo de fuga de datos,
4. define una validación correcta (sin fuga),
5. construye baselines,
6. prueba modelos candidatos razonables,
7. ajusta hiperparámetros sin tocar el test final,
8. analiza los errores,
9. evalúa por grupos clínicos,
10. entrégame un informe final en lenguaje llano.

Reglas:
- No optimices sin que la métrica esté clara y acordada conmigo.
- No uses el test final para ajustar nada.
- No aceptes una mejora sin compararla contra el baseline.
- No imputes ni escales fuera del esquema de validación.
- No uses variables que no existirían en el momento real de decidir.
- Cuando hagas un supuesto, dilo explícitamente.
- Cuando necesites mi criterio clínico, pregúntame antes de seguir.

Antes de nada, dime qué información te falta para formular bien el problema.
```

### 7.2 Prompt de exploración inicial

```text
Haz una exploración inicial orientada a un análisis clínico serio, no solo
gráficos bonitos.

Incluye:
1. estructura del dataset y qué representa cada fila,
2. distribución del objetivo (evento_cv / riesgo_cv_10a / lo que corresponda),
3. valores ausentes y su posible patrón,
4. valores atípicos o clínicamente inconsistentes,
5. relaciones preliminares con el objetivo,
6. diferencias por grupo clínico relevante,
7. posibles columnas con riesgo de fuga de datos,
8. tu recomendación para seguir con el análisis.

Para cada hallazgo, dime: qué evidencia lo respalda y qué impacto clínico tendría.
```

### 7.3 Prompt de limpieza de datos

```text
Propón un plan de limpieza conservador para este dataset.

Clasifica cada acción como:
1. segura antes de dividir en entrenamiento/test,
2. debe hacerse solo dentro del entrenamiento,
3. necesita mi confirmación clínica,
4. no recomendable.

No elimines valores atípicos por defecto.
No conviertas valores ausentes sin analizar antes si la ausencia es informativa.

Devuélveme una tabla: acción | columnas afectadas | razón clínica | riesgo | cuándo hacerlo.
```

### 7.4 Prompt de validación

```text
Diseña el esquema de validación correcto para este problema.

Considera:
- si hay varias filas por paciente,
- si hay una dimensión temporal relevante,
- el desequilibrio entre clases (si aplica),
- el tamaño real del dataset.

Devuélveme:
1. el esquema recomendado,
2. qué esquemas descartaste y por qué,
3. cómo evitar fuga de datos con este esquema,
4. cómo vas a reportar la incertidumbre entre particiones.
```

### 7.5 Prompt de análisis de errores

```text
Analiza los errores del modelo actual.

Quiero:
1. los falsos negativos con mayor confianza del modelo (los más preocupantes),
2. los falsos positivos con mayor confianza,
3. errores agrupados por grupo clínico,
4. errores asociados a valores ausentes,
5. patrones que sugieran una variable nueva a probar,
6. patrones que sugieran que la métrica elegida no es la adecuada.

Propón 3 experimentos derivados de este análisis, priorizados por impacto
clínico esperado y por riesgo de fuga de datos.
```

### 7.6 Prompt de selección final

```text
Compara los mejores modelos candidatos.

No elijas solo por la métrica principal. Evalúa también:
- estabilidad entre particiones,
- si se puede explicar a un comité clínico,
- rendimiento por grupo,
- riesgo de fuga de datos pendiente,
- facilidad para vigilar que sigue funcionando bien con el tiempo.

Recomiéndame:
1. el modelo preferido,
2. una alternativa más simple,
3. en qué condiciones elegirías la alternativa,
4. qué quedaría pendiente de revisar antes de dar por bueno el análisis.
```

### 7.7 Prompt de informe final

```text
Genera un informe final de este análisis, pensado para un profesional
sanitario sin formación en programación.

Estructura:
1. Resumen ejecutivo.
2. El problema clínico y la decisión que apoya.
3. Los datos (recuerda: SINTÉTICOS) y el objetivo.
4. Cómo se validó, sin fuga de datos.
5. Hallazgos relevantes de la exploración.
6. Modelos comparados y por qué ganó el elegido.
7. Rendimiento por grupo clínico.
8. Los errores más relevantes.
9. Limitaciones honestas.
10. Recomendación final y condiciones de uso.

El informe debe entenderse sin necesidad de leer ni una línea de código.
```

---

## 8. Lo que el agente no debe decidir solo

Aunque el agente sea potente, hay decisiones que deben quedarse siempre en manos humanas, tal como recordó la U10 en su sección 10.5:

```text
la definición exacta del objetivo clínico (target)
el coste relativo de cada tipo de error
qué variables son aceptables usar y cuáles no
qué compromiso entre métrica y explicabilidad se acepta
si el rendimiento en un grupo débil es tolerable o no
si un caso atípico es un error de registro o un hallazgo clínico real
si un modelo está listo para usarse en la práctica
```

El agente puede informar, argumentar y proponer. No debe apropiarse del criterio clínico.

---

## 9. Antipatrones frecuentes

### 9.1 "Hazme el mejor modelo"

Malo porque no define "mejor".

Mejor:

```text
Optimiza la sensibilidad en el grupo de pacientes que vamos a priorizar,
manteniendo una precisión mínima razonable. Evalúa también por grupo clínico.
```

### 9.2 "Usa todo lo que mejore la métrica"

Eso invita directamente a la fuga de datos.

Mejor:

```text
Usa solo variables disponibles antes del momento de decidir. Marca como
sospechosa cualquier variable que pueda derivarse del propio evento.
```

### 9.3 "La exploración ya está hecha, pasemos al modelo"

La exploración inicial no es un entregable cerrado: es una fuente de hipótesis.

Mejor:

```text
Por cada hallazgo de la exploración, dime qué consecuencia tiene para el
análisis o para la validación.
```

### 9.4 "Elimina los valores raros automáticamente"

Muchos valores atípicos son los casos clínicamente más importantes.

Mejor:

```text
Clasifica los valores atípicos en imposibles, errores probables, raros pero
plausibles, y casos potencialmente relevantes.
```

### 9.5 "Sigue ajustando hasta que mejore"

Esto produce sobreajuste a la validación, no una mejora real.

Mejor:

```text
Define de antemano cuándo pararías, y un conjunto de test final que solo se mira una vez.
```

### 9.6 "Aceptar el código porque funciona"

Que el código se ejecute sin error no significa que sea correcto.

Mejor:

```text
Antes de aceptar el resultado, comprueba: ¿respeta la partición por paciente?,
¿evita fuga de datos?, ¿es reproducible si lo vuelvo a ejecutar?
```

---

## 10. Dos encargos completos de ejemplo

### 10.1 Clasificación: priorización de prevención cardiovascular

```text
Actúa como mi copiloto de ciencia de datos clínicos sobre pacientes.csv
(cohorte SINTÉTICA, 20 000 pacientes, no son pacientes reales).

Contexto:
Cada fila es un paciente. Queremos priorizar a quién citar antes en el
programa de prevención cardiovascular.

Objetivo:
evento_cv = 1 si el paciente sufre un evento cardiovascular en el horizonte
de seguimiento del estudio (prevalencia aproximada 19%).

Restricción:
Solo pueden usarse variables disponibles en el momento de la consulta.
Cualquier variable derivada del propio evento debe excluirse.

Métrica principal:
sensibilidad / PR-AUC, priorizando no dejar fuera a pacientes de riesgo real.

Validación:
partición por paciente, con validación cruzada para comparar modelos.

Proceso:
1. Exploración inicial.
2. Informe de calidad de datos.
3. Informe de fuga de datos.
4. Baselines (clase mayoritaria, regresión logística).
5. Modelos candidatos (logística, Random Forest, boosting).
6. Ajuste de hiperparámetros sin tocar el test final.
7. Ajuste del umbral según el coste del falso negativo.
8. Análisis de errores.
9. Rendimiento por grupo: sexo, tabaquismo, edad, presencia de diabetes.
10. Interpretación clínica del modelo elegido.
11. Informe final en lenguaje llano.

Criterios de parada:
- prefiere el modelo más simple si la diferencia con uno más complejo es pequeña,
- para tras varios experimentos sin mejora relevante,
- para si detectas una fuga de datos que no se puede resolver con la información
  que tienes; en ese caso, pregúntame.

Entregables:
informe de exploración, informe de fuga, comparación de modelos, informe final,
resumen apto para presentar a un comité clínico.
```

### 10.2 Serie temporal: previsión de ingresos en urgencias

```text
Actúa como mi copiloto de ciencia de datos clínicos sobre urgencias_diarias.csv
(serie SINTÉTICA: fecha, ingresos, festivo, temporada_gripe, temperatura).

Unidad:
fila = día.

Objetivo:
predecir los ingresos en urgencias de los próximos 7 días, para anticipar
las necesidades de personal y camas.

Contexto:
Subestimar los ingresos puede saturar el servicio; sobreestimar de forma
sistemática hace que se sobredimensione el personal sin necesidad real.

Métrica principal:
error medio absoluto (MAE) en unidades de ingresos diarios.

Validación:
backtesting temporal (entrenar con el pasado, validar en un tramo posterior
que el modelo no ha visto). No usar particiones aleatorias.

Variables permitidas:
histórico de ingresos hasta el día anterior, calendario, festivo,
temporada_gripe, temperatura prevista.

Variables prohibidas:
ingresos del propio día o de días futuros.

Proceso:
1. Exploración temporal (tendencia, estacionalidad semanal y por temporada de
   gripe, huecos, valores atípicos).
2. Informe de fuga de datos.
3. Baselines: valor del mismo día de la semana anterior, media móvil.
4. Modelo con variables de calendario y rezagos.
5. Evaluación por horizonte (día +1, +3, +7).
6. Análisis de errores: ¿falla más en temporada de gripe?, ¿en festivos?
7. Informe final.

Criterio de parada:
el modelo debe superar claramente al baseline del "mismo día de la semana
anterior" y no mostrar un sesgo sistemático fuerte en temporada de gripe.
```

---

## 11. Cómo pedir evidencia, no opiniones

Prompts útiles para exigir rigor en cualquier momento del análisis:

```text
¿Qué evidencia respalda esta decisión?
¿Qué comprobación descartó la alternativa?
¿Qué resultado te haría cambiar de opinión?
¿Qué riesgo de fuga de datos queda pendiente?
¿Qué grupo clínico tiene un rendimiento peor?
¿Qué parte de este resultado es inestable entre particiones?
¿Qué supuestos has hecho sin decírmelo?
¿Qué debería revisar yo antes de confiar en esto?
```

El agente debe trabajar dejando un rastro que se pueda seguir. Un resultado bien argumentado no es:

```text
"El modelo X es el mejor."
```

Es:

```text
"El modelo X tiene la mejor sensibilidad media, pero varía bastante entre
particiones. El modelo Y es algo peor en sensibilidad, pero mucho más estable
y fácil de explicar a un comité clínico. Recomiendo X si prioriza la métrica
pura; Y si prioriza la confianza clínica en el resultado."
```

Eso es criterio, y es exactamente lo que la sección 10.5 de la U10 pidió como "cuestionario clínico" antes de fiarte del agente: exígele siempre la evidencia ejecutada, nunca una afirmación con aplomo pero sin número real detrás.

---

## Qué llevarte

* **Antes de analizar, define los seis contratos**: problema, datos, objetivo, métrica, validación y uso clínico. Un problema bien formalizado es más importante que cualquier modelo.
* **El contexto es el combustible**: dile al agente qué representa cada fila, qué columnas son "del futuro", qué error es más caro y qué significa éxito. Sin eso, el agente hace algo genérico y razonable en abstracto, no algo dirigido a tu caso.
* **El bucle de evidencia se repite en cada fase**: proponer, ejecutar, medir, interpretar y validar clínicamente, desde la exploración inicial hasta el informe final. Ninguna fase se salta la validación humana.
* **Hay decisiones que nunca delegas**: la definición del objetivo, el coste de los errores, qué variables son aceptables y si un modelo está listo para usarse. El agente informa; tú decides.
* **Pide siempre evidencia, no opiniones**: un número dicho con seguridad no es un número medido. Exige la comparación contra un baseline, la variación entre particiones y el rendimiento por grupo clínico antes de dar nada por bueno.

***

Con esta guía tienes en un solo sitio lo que necesitas para dirigir tú mismo un análisis clínico completo, de la primera pregunta al informe final. Si quieres profundizar en cómo un agente ajusta y mejora su propia búsqueda dentro de una fase concreta —por ejemplo, cómo decide qué probar a continuación durante el ajuste de hiperparámetros—, el estudio [**Bucles de auto-mejora**](prompting-loops.md) mira justo ese mecanismo con más detalle.
