---
description: >-
  El nuevo oficio en 2026: no programas el modelo, diriges un bucle. Le das a un
  agente los datos y el contexto clínico, y él formula experimentos, ejecuta
  código, lee resultados y decide qué mejorar. Prompting, metaprompting, el bucle
  de evidencia con el clínico al mando, y las herramientas (Cowork, Claude Code,
  Codex, Manus) explicadas en modo idea. Todos los datos son sintéticos.
---

# U10 · La IA como copiloto de ciencia de datos

Todo el curso ha empujado hacia esta unidad. Desde la primera página defendemos una idea que a un profesional sanitario le cambia la vida: en 2026 **el valor no está en escribir `scikit-learn` de memoria, sino en tener criterio clínico y saber pedir el código a un asistente de IA**.

Hemos visto qué hace cada familia de modelos (U4–U8), cómo evaluar con honestidad (U3), cómo limpiar los datos (U2) y dónde viven los modelos ya hechos (U9). Ahora damos el último paso y el más importante: aprender a **dirigir** todo eso de principio a fin, convirtiendo a la IA en tu **copiloto de ciencia de datos**.

Y aquí hay una novedad genuina, no un truco más. Hasta hace poco, un asistente "escribía un trozo de código" y tú lo pegabas en Colab. Los agentes de 2026 hacen algo cualitativamente distinto: **toman procesos enteros**.

Les das los **datos** y el **contexto clínico**, y el agente —apoyado en su enorme conocimiento de estadística y de ciencia de datos— empieza a **formular experimentos, ejecutarlos en Python, leer los resultados, interpretarlos y decidir qué cambiar para mejorar**. Y vuelve a empezar. A esa forma de trabajar se la empieza a llamar **loop engineering**: no programas el modelo, **diseñas y diriges un bucle**.

{% hint style="success" %}
**💡 Idea clave**

El cambio de oficio se resume en una frase: **antes escribías el pipeline; ahora diriges un bucle de evidencia**. Tú aportas la **pregunta clínica correcta**, el **contexto** y el **juicio**; el agente aporta **velocidad y amplitud** —prueba muchas más cosas, mucho más rápido, de las que probarías a mano—. Lo que **no** delegas nunca es la validación: en salud, cada vuelta del bucle la valida un criterio clínico, porque un modelo plausible pero erróneo no es un fallo de código, es un riesgo para un paciente.
{% endhint %}

Los fundamentos del curso no valen ahora menos: valen **más**. La métrica clínica adecuada (U3), la partición por paciente sin fugas (U3, U5, U7), la lección de "empieza por lo simple" (U4–U5)… son exactamente el **criterio con el que juzgas lo que el agente te trae**. Un agente genera código bueno muy rápido y también código plausible-pero-equivocado muy rápido; tu experiencia clínica es lo que distingue uno de otro.

### Objetivos de esta unidad

* Entender el **cambio de oficio**: de escribir el modelo a **dirigir un bucle** (*loop engineering*), con el clínico al mando.
* Aprender a **pedir bien** a un asistente para ciencia de datos: dar **contexto, objetivo, restricciones, métrica clínica y criterio de éxito** (*prompting*), y pedirle que **mejore tu propio prompt** (*metaprompting*).
* Interiorizar el **bucle de evidencia** —proponer → ejecutar → medir → interpretar → repetir— con validación clínica en cada vuelta, aplicado a un caso real: de `pacientes.csv` a un modelo de `evento_cv` evaluado.
* Conocer, **en modo idea** (qué implican, no un tutorial), las herramientas que encarnan este modo de trabajo: **Cowork**, **Claude Code**, **Codex** y **Manus**.
* Fijar el **papel del sanitario** frente al del agente, con sus **riesgos** propios (alucinaciones, resultados plausibles pero falsos, fugas de datos que el agente no ve) y por qué **el humano valida**.
* Llevarte una **plantilla de prompt reutilizable** para análisis clínico.

Como en todo el curso, los ejemplos se apoyan en la cohorte **sintética** `pacientes.csv` (20 000 pacientes generados por código, **no son pacientes reales**), con sus dos objetivos: `riesgo_cv_10a` (riesgo cardiovascular a 10 años, en %, para **regresión**) y `evento_cv` (0/1, para **clasificación**, con **prevalencia ≈ 19 %**).

## 10.1 El nuevo oficio: dirigir un bucle, no escribir el modelo

Conviene ver con claridad qué cambia. Durante décadas, un proyecto de ciencia de datos era un **guion lineal** que recorrías paso a paso: exploras los datos, los limpias, construyes variables, entrenas modelos, ajustas, evalúas y redactas el informe. Cada paso lo escribías tú, a mano, y avanzabas en línea recta.

El trabajo con agentes se parece más a un **bucle de evidencia**: tú pones el **objetivo, el contexto y el criterio**; el agente **ejecuta e itera**; y tú **decides**. La unidad de trabajo deja de ser "una línea de código" y pasa a ser "un experimento con su resultado medido". No es que el modelo se reinvente solo: es una **búsqueda guiada por razonamiento**, con la **realidad como árbitro** —una métrica ejecutada de verdad, no la opinión del modelo—.

{% hint style="info" %}
**Concepto · Loop engineering (dirigir un bucle de evidencia)**

Forma de trabajar en la que no escribes el pipeline paso a paso, sino que **defines un objetivo, das contexto y fijas un criterio**, y un agente **formula experimentos, ejecuta código, mide una métrica real, interpreta el resultado y decide el siguiente paso**, iterando. El humano no picotea código: **gobierna el bucle** —revisa la evidencia y valida las decisiones que importan—.
{% endhint %}

La diferencia decisiva frente a "un chat que escribe código" es que **el agente ejecuta**. No te devuelve un texto con código para que lo copies: **lo pone a funcionar** en un entorno real, obtiene los números (la métrica, los errores, los gráficos) y usa ese **feedback real** —no uno imaginado— para decidir qué probar a continuación.

Sin ejecución no hay bucle de evidencia: solo habría opiniones encadenadas. Esa capacidad de **cerrar el ciclo** —proponer, ejecutar, medir, corregir— es lo que separa a un agente de un chatbot que "sabe" de estadística pero no ha corrido nada.

Para un profesional sanitario, esto es liberador. Ya no necesitas dominar la sintaxis para hacer un análisis serio: necesitas **plantear bien la pregunta clínica** y **saber leer la evidencia**. El agente cubre la brecha técnica; tú aportas lo que ningún modelo tiene: el **contexto del paciente y de la práctica clínica**.

## 10.2 Prompting para ciencia de datos: cómo pedir bien

Si el oficio nuevo es dirigir, la primera destreza es **pedir bien**. Y "pedir bien" no es escribir un ensayo: es **dar el contexto correcto**.

Un agente sin contexto es como un residente brillantísimo que **no conoce tu caso**: hará algo genérico y razonable en abstracto, probablemente con errores sutiles que a ti te saltarían a la vista. El **mismo** agente, con buen contexto, se comporta como un compañero senior que **sí** conoce el problema. El contexto es lo que hace que el modelo **inyecte su conocimiento en la dirección clínica correcta**.

La diferencia se ve de un vistazo:

```
❌ Prompt pobre:
"Te paso pacientes.csv, hazme el mejor modelo."

✅ Prompt con contexto clínico:
"Te paso pacientes.csv (cohorte SINTÉTICA, no son pacientes reales). Quiero predecir
evento_cv (evento cardiovascular, prevalencia ≈19%) para ayudar a priorizar a quién
citar antes en prevención. Nos duele mucho más NO detectar a un paciente que tendrá
el evento (falso negativo) que citar de más a uno sano. Quiero una PROBABILIDAD bien
calibrada, no solo un sí/no. Valida por paciente, sin fuga de datos. Y ojo: hay
columnas que en la práctica no tendría en el momento de decidir a quién cito."
```

El segundo prompt no es "más largo por presumir": cada frase cierra una puerta a un error. Un buen prompt de ciencia de datos responde, en el fondo, a un puñado de **preguntas** que tú —como clínico— sabes contestar mejor que nadie:

* **Contexto y objetivo** — ¿qué **decisión clínica** se toma con la predicción?, ¿quién la usa?, ¿para qué sirve? *(Priorizar citas de prevención; la usa el equipo de atención primaria.)*
* **Los datos** — ¿qué representa **una fila**? (un paciente, una visita, un día…), ¿cuál es la clave?, ¿qué columnas son fechas, identificadores o texto?, y sobre todo **¿cuáles solo existen DESPUÉS del evento?** *(Una fila = un paciente; la fecha del evento no la tendría al decidir.)*
* **El objetivo a predecir (target)** — ¿cómo se define la variable respuesta?, ¿en qué ventana temporal?, ¿la observamos o la inferimos? *(`evento_cv` = 1 si ocurre el evento en el horizonte de seguimiento.)*
* **La métrica clínica** — la brújula, y debe **conectar con la decisión**. Aquí no vale la *accuracy* (con prevalencia ≈19 %, "digo que nadie tendrá evento" acierta el 81 % y es inútil). Si el falso negativo es lo caro, la **sensibilidad/recall**, el **PR-AUC** o la **sensibilidad en el grupo que voy a citar** hablan tu idioma (todo esto es U3).
* **La validación** — el reparto entrenamiento/test debe **imitar la práctica real**: **por paciente** si hay varias filas por persona, **temporal** si hay tiempo (U7). Regla de oro: todo lo que el pipeline aprenda de los datos (imputar, escalar, codificar) se aprende **solo dentro del train**, nunca mirando el test.
* **El criterio de éxito y las restricciones** — ¿cuándo consideras que "ya está bien"?, ¿hace falta **explicar** cada predicción (SHAP, U5)?, ¿qué está **prohibido** usar?

{% hint style="warning" %}
**⚠️ Aviso · La métrica mal elegida hace correr al agente en la dirección equivocada**

Si le das al agente una métrica que no refleja tu decisión clínica, optimizará **con diligencia hacia el sitio equivocado**. Un modelo con *accuracy* del 88 % sobre `evento_cv` puede estar **sin detectar** a casi ningún paciente que sí sufrirá el evento —precisamente los que te importan—. Elegir la métrica es una **decisión clínica**, no un detalle técnico: es lo primero que debes fijar (o revisar si el agente la propone).
{% endhint %}

### Metaprompting: pídele al asistente que mejore tu propio prompt

Aquí hay un truco tan simple como potente. Si no estás seguro de haber pedido bien, **pídeselo al propio asistente**: que **revise y mejore tu prompt** antes de ejecutarlo.

A eso se le llama **metaprompting**, y para un no programador es una red de seguridad excelente, porque el modelo conoce los fallos habituales de un encargo de ciencia de datos (métrica ausente, fuga de datos, ambigüedad en el target) mejor que un principiante.

{% hint style="info" %}
**Concepto · Metaprompting**

Pedirle al asistente que **mejore tu propia instrucción**: que detecte lo que falta (contexto, métrica, validación, restricciones), te haga las preguntas que necesita y te devuelva un prompt más completo y sin ambigüedades, **antes** de ejecutar nada. Es "usar la IA para aprender a pedirle mejor a la IA".
{% endhint %}

En la práctica, un buen prompt de metaprompting suena así:

```
Antes de ejecutar nada, actúa como un experto en ciencia de datos clínicos y
CRITICA este encargo mío. Dime:
1. Qué información CLAVE me falta para que el resultado sea fiable (target exacto,
   qué columnas existen en el momento de decidir, coste de cada tipo de error).
2. Si la métrica que propongo encaja con mi decisión clínica; si no, sugiéreme una
   mejor y explícame por qué.
3. Qué riesgos de fuga de datos (leakage) ves en mi planteamiento.
Luego reescríbeme el prompt ya corregido y pregúntame lo que aún necesites.

--- Mi encargo ---
[aquí pego mi prompt inicial]
```

Fíjate en el cambio de actitud: no le pides que **obedezca**, le pides que te **ayude a pensar**. Es exactamente lo que haría un buen compañero senior antes de dejarte lanzar un análisis.

{% hint style="success" %}
**💡 Idea clave**

**Metaprompting** es, en el fondo, delegar en la IA la revisión de tu propio encargo antes de gastar ni un experimento. Cuesta una vuelta extra de conversación y evita horas de bucle mal dirigido: barato asegurar que el objetivo, la métrica y las restricciones están bien puestos **antes** de que el agente empiece a ejecutar.
{% endhint %}

## 10.3 El bucle de evidencia, paso a paso

El corazón del *loop engineering* es un ciclo sencillo que se repite. En cada vuelta el agente **propone** (qué va a probar y por qué), **ejecuta** (escribe y corre código real), **mide** (una métrica de verdad), **interpreta** (qué significa ese número clínicamente) y **decide** si conserva o descarta el cambio. Y vuelve a empezar. Lo que lo convierte en mejora real —y no en charla— es que la observación es una **métrica objetiva ejecutada**, no una impresión.

En salud, ese bucle tiene una **quinta pieza obligatoria en cada vuelta: la validación clínica**. El humano no espera al final: revisa que la decisión del agente tenga **sentido clínico**, que no haya metido una variable imposible, que la métrica siga siendo la correcta. Es el paso de **_human in the loop_** (apruebas cada paso importante) o, más habitual, **_human on the loop_** (supervisas desde arriba y el agente se detiene en las decisiones clave que él mismo señala).

{% hint style="info" %}
**Concepto · _Human in the loop_ vs _human on the loop_**

*In the loop*: el humano **aprueba cada paso** (control fino, más lento). *On the loop*: el agente ejecuta el bucle y el humano **supervisa desde arriba**, interviniendo solo en las decisiones clave (métrica final, exclusión de una variable, modelo elegido, umbral). En salud, la tendencia a dar más autonomía al agente **nunca** significa renunciar al criterio ni al gobierno clínico: significa reservar tu atención para lo que de verdad importa.
{% endhint %}

Veámoslo aplicado a un caso concreto que **une U2–U5**: partimos de `pacientes.csv` y queremos un modelo de `evento_cv` **evaluado con honestidad**, eligiendo el mejor entre varios candidatos, con validación cruzada e hiperparámetros. Así se desarrollaría el bucle, vuelta a vuelta:

1. **Proponer.** El agente formaliza el problema y detecta ambigüedades: *"¿el target es evento en qué horizonte?, ¿qué columnas no tendrías al decidir?"*. Propone **partición por paciente**, **métrica orientada a no perder casos** (recall/PR-AUC) y un plan: baseline → logística → Random Forest → boosting.
2. **Ejecutar.** Escribe y **corre** el código: imputa y escala **solo en el train**, monta la **validación cruzada k-fold**, entrena los candidatos. No teoriza: ejecuta.
3. **Medir.** Obtiene números reales. En nuestra cohorte sintética, la **regresión logística** alcanza **AUC ≈ 0,84** en `evento_cv` y **gana a Random Forest** (≈ 0,83), porque el riesgo es aproximadamente **log-aditivo** —la gran lección de "empieza por lo simple" (U4–U5)—.
4. **Interpretar (con validación clínica).** El agente te lo explica y **tú lo validas**: *"lo simple gana, y además es interpretable; el gradiente por tabaquismo (nunca ≈14 %, ex ≈22 %, activo ≈28 %) y el efecto protector del HDL y la actividad física son clínicamente coherentes"*. Aquí es donde tu criterio confirma —o corrige— la lectura.
5. **Repetir.** Con la logística como favorita, ajusta hiperparámetros **sin tocar el test final**, revisa la **calibración** (U3) y el umbral según el coste del falso negativo. Cuando varias vueltas no mejoran de forma relevante, el agente propone **congelar**, evaluar **una sola vez** en el holdout y entregar informe + *model card*.

El desenlace es la moraleja de todo el curso: **el mejor modelo no se adivina ni se elige por moda, se mide** —y aquí el agente lo ha medido por ti, pero **tú** has validado que midiera lo correcto—.

Que la humilde logística gane a un Random Forest para `evento_cv` (mientras que, recuérdalo, para la **regresión** de `riesgo_cv_10a` es al revés: Random Forest **R² ≈ 0,91** supera a la lineal **R² ≈ 0,81** por las interacciones) es justo el tipo de hallazgo que un buen bucle de evidencia saca a la luz.

{% hint style="success" %}
**💡 Idea clave**

El agente **te puede sorprender, y para bien**. Tu preconcepción del problema es un excelente punto de partida, pero cuando el agente **explora todas las decisiones del pipeline** —métrica, validación, variables, modelo— a veces llega a una solución que **mejora tu intuición**. Ejemplo: le fijas `f1` como métrica; el agente, viendo que la clase positiva es rara y que solo vas a citar a un grupo reducido, te responde: *"`f1` no refleja bien tu decisión; propongo optimizar la sensibilidad en el grupo que vas a citar y ajustar el umbral, ¿te parece?"*. No te lo impone: te lo **argumenta**, y **decides tú**. Aprovechar esa capacidad exploratoria es, cada vez más, de donde salen las mejores soluciones.
{% endhint %}

## 10.4 Las herramientas, en modo idea

Conviene poner nombre a las herramientas que encarnan este modo de trabajar, pero con una advertencia: aquí las tratamos **en modo idea** —qué implican y para qué sirven—, **no como un tutorial**.

Cambian rápido y no hace falta dominar ninguna para entender el concepto. Lo que las une es que **van más allá de "un chat que escribe código"**: pueden **tomar procesos enteros, ejecutar en bucle y entregar resultados** (código que corre, informes, un análisis terminado), no solo un mensaje con texto.

{% hint style="info" %}
**Concepto · Agente (frente a un chatbot)**

Un **chatbot** conversa y, como mucho, te devuelve código para que lo ejecutes tú. Un **agente** vive en un entorno donde puede **ejecutar** ese código, ver el resultado real y **decidir el siguiente paso** él mismo, repitiendo el ciclo hasta cumplir un objetivo. La diferencia no es "sabe más", sino que **cierra el bucle**: propone, actúa, mide y corrige.
{% endhint %}

Estas son las cuatro que conviene conocer:

* **Cowork** — es **Claude trabajando en tu escritorio**, con tus **ficheros y carpetas**. En vez de copiar y pegar un CSV en un chat, le señalas una carpeta ("aquí tengo `pacientes.csv` y mis notas") y trabaja **sobre tus documentos** directamente: los lee, produce análisis, deja resultados en tu propio ordenador. La idea clave: **acerca el agente a donde ya viven tus datos y tu material**, con una interacción cercana al escritorio de siempre.
* **Claude Code** — un **agente que vive en el terminal o el editor**, pensado para **tareas de código y análisis**. Puede recorrer un proyecto entero, escribir y ejecutar programas, correr el bucle de experimentos y dejarte los artefactos (scripts, notebooks, informes). Es la herramienta cuando el trabajo es **eminentemente de código** y quieres al agente operando con soltura en ese entorno.
* **Codex** — **muy similar a Claude Code en concepto**: un agente de código en el terminal/editor que toma tareas técnicas, ejecuta y entrega. Que existan varias opciones equivalentes es buena señal —es una categoría, no un producto único—; para nuestros fines, Claude Code y Codex representan **la misma idea**: un agente de código que abre y cierra el bucle.
* **Manus** — un **agente autónomo**: le das un objetivo y trabaja **por su cuenta** durante un rato, encadenando pasos (buscar, ejecutar, redactar) hasta entregarte un resultado. Encarna el extremo más **autónomo** del espectro: menos conversación paso a paso, más "vuelve cuando esté hecho".

¿En qué se parecen y en qué se diferencian de "un chat que escribe código"? Se parecen en que todos **entienden y generan** código a partir de tu intención. Se diferencian en lo esencial: un chat **te entrega texto**; estos agentes **toman el proceso, lo ejecutan en bucle y te devuelven un resultado terminado**. Entre ellos, la diferencia es sobre todo de **dónde operan** (tu escritorio con Cowork; el terminal/editor con Claude Code y Codex) y de **cuánta autonomía** asumen (Manus, en el extremo autónomo).

{% hint style="warning" %}
**⚠️ Aviso · Autonomía, sin exagerar**

Que un agente ejecute en bucle **no** significa que sea infalible ni que trabaje sin supervisión. Optimiza hacia el objetivo y la métrica que **tú** le das: si están mal planteados, corre con diligencia hacia el sitio equivocado. La autonomía es **una herramienta de velocidad, no una delegación del juicio**. En salud, cuanto más autónomo es el bucle, **más importa** que el objetivo, la métrica y las restricciones estén bien definidos, y que un humano valide el resultado antes de darle ningún uso.
{% endhint %}

## 10.5 El papel del sanitario (y por qué el humano valida)

Con agentes tan capaces, es tentador pensar que el profesional sobra. Es justo al revés. El reparto de papeles es nítido y **complementario**:

* **El sanitario aporta lo que el agente no tiene**: la **pregunta correcta**, el **contexto clínico** (qué significa cada variable, qué error cuesta más, qué es plausible y qué no) y el **juicio** para validar la evidencia. Nadie mejor que tú sabe que una glucemia de 400 mg/dL en reposo es sospechosa, que la fecha del evento no puede usarse para predecirlo, o que un modelo que ignora el tabaquismo es clínicamente inverosímil.
* **El agente aporta velocidad y amplitud**: prueba muchas más combinaciones —variables, modelos, umbrales— y mucho más rápido de lo que harías a mano, y ejecuta el trabajo mecánico sin fatiga.

La razón de fondo por la que **el humano valida en cada vuelta** son tres riesgos propios de estos sistemas, que ningún agente detecta solo:

* **Alucinaciones.** El modelo puede **afirmar con total aplomo** algo falso: citar una métrica que no calculó, "recordar" un resultado que no ejecutó o inventar una referencia. Un número dicho con seguridad **no es un número medido**; exígele siempre la evidencia ejecutada.
* **Resultados plausibles pero erróneos.** Es el riesgo más peligroso porque **no chirría**. Un AUC de 0,97 puede parecer excelente y ser el síntoma de una **fuga de datos** (una variable que "sabe" el futuro). En clínica, un modelo plausible-pero-falso es más peligroso que uno obviamente roto, porque **te lo crees**.
* **Fugas de datos que el agente no ve.** El agente no conoce tu **flujo asistencial**: no sabe que cierta columna se rellena **después** de que ocurra el evento, o que un identificador de centro está **correlacionado con el desenlace** por un motivo organizativo. Esa fuga es **invisible en la tabla** y solo evidente **con tu conocimiento del proceso**. Por eso tú, y no el agente, eres la última defensa.

{% hint style="warning" %}
**⚠️ Aviso · Antes de fiarte del agente, pásale tu cuestionario clínico**

Ante cualquier resultado que te traiga el bucle, pregúntate (y pregúntale): ¿de dónde salen estas columnas y **existían en el momento de decidir**? ¿La partición imita la práctica real (por paciente, temporal)? ¿La **métrica** refleja mi decisión clínica? ¿Un resultado "demasiado bueno" no será **fuga de datos**? ¿Está **calibrado** (U3)? ¿Los efectos tienen **sentido clínico**? Y, la de siempre en este curso: **por una API pública, solo datos sintéticos o anonimizados** (U9). Ese cuestionario es tu criterio; el agente ejecuta, pero **tú firmas**.
{% endhint %}

## 10.6 Una plantilla de prompt reutilizable para análisis clínico

Reuniéndolo todo, esta es una **plantilla de arranque** que puedes copiar, rellenar y reutilizar en cualquier análisis. Pone al agente en su papel de copiloto, le da las reglas del juego clínicas y te reserva a ti el gobierno del bucle. Cambia lo que está entre `[...]` por tu caso.

```
Actúa como mi copiloto de ciencia de datos clínicos y trabajaremos EN BUCLE.
Conduces tú la ejecución; yo, como profesional sanitario, superviso y valido.

CONTEXTO
· Datos: [p. ej. pacientes.csv — cohorte SINTÉTICA, no son pacientes reales].
· Una fila representa: [p. ej. un paciente]. Clave: [p. ej. paciente_id].
· Decisión clínica que se tomará con la predicción: [p. ej. priorizar a quién citar
  antes en prevención].
· Objetivo a predecir (target): [p. ej. evento_cv, 0/1, prevalencia ≈19%], definido
  como [ventana / cómo se observa].
· Coste de los errores: me duele más [p. ej. un falso negativo] que [un falso positivo].
· PROHIBIDO usar (no disponible al decidir / fuga de datos): [columnas posteriores al
  evento, identificadores sospechosos…].

LO QUE TE PIDO
1. METAPROMPTING primero: dime qué información CLAVE me falta y si la métrica encaja
   con mi decisión; propónme TÚ la métrica y la validación más adecuadas y JUSTIFÍCALAS.
2. Luego repite este BUCLE hasta que propongas parar:
   · PROPÓN  — la hipótesis o el paso que vas a probar y por qué.
   · EJECUTA — escribe y CORRE el código (imputa/escala SOLO en train); si falta una
     librería, instálala; si te falta una decisión clínica, PREGÚNTAME.
   · MIDE    — con la métrica acordada; enséñame el número o el gráfico REAL.
   · INTERPRETA — qué significa clínicamente; ¿tiene sentido?
   · DECIDE  — ¿conservas o descartas? Una frase de porqué y a la siguiente vuelta.
3. DETENTE a pedir mi visto bueno en las decisiones importantes: métrica final,
   exclusión de una variable, modelo elegido, umbral.

REGLAS DE ORO (clínicas)
· Empieza por un baseline y por lo simple; no aceptes una mejora sin compararla.
· No uses el test final para ajustar; evalúalo UNA sola vez, al final.
· Nada de fuga de datos; todo el preprocesado, dentro del pipeline.
· Un resultado "demasiado bueno" es sospechoso: búscame la fuga antes de celebrar.
· Cada afirmación, respaldada por una métrica EJECUTADA (no por tu impresión).
· Si crees que el problema está mal planteado o que el modelo NO debería usarse en
  clínica, dímelo con argumentos.

Empieza por tus preguntas y por proponerme la métrica y la validación.
```

La misma estructura da versiones más específicas: un prompt solo para el **EDA y la detección de problemas de calidad** (U2), otro para el **análisis de errores**, otro para el **informe final**. El patrón común no cambia: **pide evidencia y una tabla de decisiones, no opiniones**, y reserva para ti las decisiones que de verdad importan.

## 10.7 Práctica en Colab

{% hint style="success" %}
**🔬 Práctica en Colab** — `U10_Copiloto_DataScience.ipynb` · [Abrir en Colab](https://colab.research.google.com/drive/1Nz_xdzwz0SX_fOEbzSl0jbVLVHtUnu5H)

**Un recorrido end-to-end dirigido por prompting y por bucles.** El notebook reproduce el caso de la sección 10.3 —de `pacientes.csv` a un modelo de `evento_cv` evaluado— pero **narrado como una conversación de bucle de evidencia**: cada bloque muestra el **prompt** que le darías al copiloto, el **código que ejecutaría**, el **número real** que devuelve y la **validación clínica** de esa vuelta. Se recorre metaprompting → propuesta de métrica y validación → baseline → logística vs. Random Forest vs. boosting → validación cruzada e hiperparámetros → elección del mejor modelo → holdout final. La cohorte es **sintética** y su **primera celda genera los datos**, así que no hay que descargar nada: se abre y se ejecuta. El objetivo no es aprender a teclear, sino **sentir cómo se dirige un bucle** y dónde entra tu criterio clínico en cada vuelta.
{% endhint %}

**🤖 Prompt para el asistente · Arrancar el bucle sobre `pacientes.csv`**

```
Con 'pacientes.csv' (cohorte SINTÉTICA; target de clasificación: evento_cv,
prevalencia ≈19%), en español y por celdas, actúa como mi copiloto de ciencia de
datos y trabaja EN BUCLE:
1. Primero critica mi encargo (metaprompting): dime qué te falta y propónme la
   métrica y la validación (por paciente, sin fuga) con su porqué.
2. Construye un baseline y una regresión logística; mídelas con AUC y con la
   sensibilidad en el grupo que citaría; enséñame la curva de calibración.
3. Prueba Random Forest y un boosting; compáralos con la logística en validación
   cruzada. Dime qué gana y por qué (¿el riesgo es log-aditivo?).
4. Interpreta clínicamente: efecto del tabaquismo (nunca/ex/activo), del HDL y de
   la actividad física. ¿Tiene sentido?
5. Con el mejor modelo, ajusta hiperparámetros SIN tocar el test; elige umbral
   según el coste del falso negativo; evalúa UNA vez en el holdout y entrégame un
   mini-informe con las decisiones y su porqué.
Párate a pedir mi visto bueno en la métrica final, el modelo elegido y el umbral.
```

*Fíjate en el punto 1: no le pides que empiece a programar, le pides que primero **piense contigo** y proponga la métrica. Ese pequeño gesto —metaprompting antes de ejecutar— es lo que convierte un encargo ingenuo en un bucle de evidencia dirigido con criterio.*

## Qué llevarte

* **El oficio cambió: diriges un bucle, no escribes el modelo.** Das **datos + contexto clínico + criterio**; el agente **formula, ejecuta, mide e interpreta**, y tú **validas y decides**. A eso se le llama **loop engineering**, y para un no programador es lo que hace posible un análisis serio.
* **Pedir bien es dar contexto.** Objetivo, datos (qué es una fila, qué columnas son "del futuro"), target, **métrica clínica**, validación sin fuga y criterio de éxito. Y si dudas, **metaprompting**: pídele al asistente que **mejore tu propio prompt** antes de ejecutar.
* **El bucle de evidencia lleva una quinta pieza en salud: la validación clínica en cada vuelta.** Proponer → ejecutar → medir → interpretar → **validar** → repetir, con el humano *on the loop*. Aplicado a `pacientes.csv`, reencuentras la lección del curso: la logística **gana** en `evento_cv` (AUC ≈ 0,84) porque el riesgo es log-aditivo; **se mide, no se adivina**.
* **Las herramientas, en modo idea.** **Cowork** (Claude sobre tus ficheros y carpetas), **Claude Code** y **Codex** (agentes de código en el terminal/editor) y **Manus** (agente autónomo) comparten una idea: **toman procesos enteros, ejecutan en bucle y entregan resultados**, no solo texto. Difieren en **dónde operan** y en **cuánta autonomía** asumen —sin exagerarla nunca—.
* **Tu papel es insustituible.** Aportas la pregunta correcta, el contexto y el juicio; el agente, velocidad y amplitud. Y **el humano valida** porque hay riesgos que el agente no ve: **alucinaciones**, **resultados plausibles pero falsos** y **fugas de datos** invisibles en la tabla y solo evidentes con tu conocimiento del proceso asistencial.

***

Con esta unidad cierras el arco del curso: sabes **qué** modelo encaja en cada problema clínico (U2–U8), **dónde** están los modelos ya hechos (U9) y ahora **cómo dirigir** todo el proceso como un copiloto de ciencia de datos. Solo queda una pieza, transversal y decisiva, que ha aparecido una y otra vez —la fuga de datos que no debes cometer, el "solo datos sintéticos", el sesgo que hay que vigilar—: reunir de forma ordenada la **ética, el sesgo, la validación clínica y la privacidad**. Es el terreno de la **Unidad 11**, y es lo que convierte todo lo aprendido en una práctica **responsable**.
