---
description: >-
  ¿Basta con pedirle el código a un asistente en un chat, o hace falta un
  agente que itere solo? Niveles de automatización para afinar un modelo de
  riesgo cardiovascular, el patrón de contexto+objetivo+métrica+criterio de
  éxito, y cómo el sanitario supervisa cada vuelta. Estudio complementario.
---

# Estudio · Prompting, metaprompting y loop engineering para afinar un modelo

{% hint style="info" %}
**Documento hermano.** Este estudio profundiza en un caso muy concreto: **el bucle de auto-mejora aplicado a elegir y afinar hiperparámetros**. Si quieres primero el marco general de "dirigir un bucle" —prompting, metaprompting, el bucle de evidencia completo, las herramientas (Cowork, Claude Code, Codex, Manus) y el papel del sanitario— consulta la [**Unidad 10 · La IA como copiloto de ciencia de datos**](../unidades/u10-copiloto-datascience.md). Y si buscas el recorrido completo de un proyecto de ML con agentes de principio a fin, mira el estudio [**Copiloto de ciencia de datos**](copiloto-datascience.md). Este documento asume esos dos y se centra en un rincón muy concreto: **¿cómo se ve, en la práctica, un bucle que busca los mejores hiperparámetros solo?**
{% endhint %}

Este es un documento de **estudio independiente**, no una unidad más del temario. Nace de una pregunta muy concreta que aparece en cuanto se domina la Unidad 5: allí aprendimos que un modelo tiene "mandos" que hay que ajustar antes de entrenar —los **hiperparámetros**— y que buscarlos bien es casi una búsqueda a ciegas (`GridSearchCV`, `RandomizedSearchCV`). ¿Puede un asistente de IA ayudarnos con esa búsqueda de forma más activa que "generar el código una vez"? ¿Puede, literalmente, **iterar solo**: proponer una configuración, ejecutarla, mirar el resultado y proponer la siguiente, sin que nosotros hagamos de mensajero en cada vuelta?

La respuesta es sí, y hay varias formas de hacerlo, con más o menos automatización. A eso lo llamamos **loop engineering**: en lugar de escribir el pipeline de búsqueda a mano, tú **diriges un bucle** que otro (el agente) ejecuta. Todo el documento se apoya en el mismo caso que ya conoces: afinar un modelo que predice `evento_cv` sobre la cohorte **sintética** [`pacientes.csv`](https://drive.google.com/file/d/1Ku0j-sAf8Cr3FPT-DGm8v5p4h_2BmV5U/view?usp=drive_link) (20 000 pacientes generados por código, **no son pacientes reales**).

{% hint style="success" %}
**💡 Idea clave**

La búsqueda de hiperparámetros deja de ser solo un algoritmo numérico (rejilla, aleatoria) para convertirse, opcionalmente, en un **proceso de razonamiento iterativo**: el propio asistente lee el histórico de intentos, razona sobre qué probar a continuación y te devuelve una versión mejorada del código. No sustituye a lo que aprendiste en la Unidad 5; lo **complementa**, y en algunos casos lo **orquesta**.
{% endhint %}

## 1. Qué es un bucle de auto-mejora

Un bucle de auto-mejora tiene tres fases que se repiten: **Proponer → Ejecutar → Observar**. El asistente decide qué probar a continuación (proponer), lo pone a funcionar de verdad —escribe y corre código en Colab, no solo lo describe— (ejecutar), y lee el resultado real de esa ejecución para decidir el siguiente paso (observar). Lo que convierte esto en "auto-mejora" y no en una charla es que la observación es una **métrica objetiva** medida de verdad, y existe una **regla de decisión** clara: si la métrica mejora, se conserva el cambio; si empeora, se descarta. Y vuelta a empezar.

{% hint style="info" %}
**Concepto · Bucle de auto-mejora (agentic loop)**

Ciclo iterativo en el que un asistente de IA propone una acción —por ejemplo, una configuración de hiperparámetros—, la ejecuta en un entorno real (Colab), observa un resultado medible (una métrica) y decide el siguiente paso, repitiendo hasta cumplir un criterio de parada. La diferencia con "pedirle el código una vez" es que el asistente **cierra el ciclo con la realidad**: ejecuta y mide, en lugar de limitarse a proponer y confiar en su propia intuición.
{% endhint %}

Un ejemplo ilustrativo ayuda a fijar la idea, aunque no sea clínico. En marzo de 2026, el investigador Andrej Karpathy publicó un proyecto minimalista que dejaba a un asistente **editando un único fichero de entrenamiento**, ejecutando un experimento acotado en tiempo, midiendo si una métrica mejoraba, y conservando o revirtiendo el cambio; repitiendo esto toda la noche sin intervención humana llegó a docenas de mejoras encadenadas. La receta que popularizó es exactamente la que necesitamos para hiperparámetros: **un único fichero que se puede modificar, una única métrica, un presupuesto fijo de intentos, y una regla de conservar o descartar**.

{% hint style="success" %}
**💡 Idea clave**

Para nuestro caso, esa receta se traduce así: **un solo notebook · el AUC (o el recall) de `evento_cv` · un número fijo de vueltas · conservar si mejora, descartar si no**. Sobre esa estructura tan simple se puede construir desde un bucle completamente manual hasta un asistente que itera casi sin ayuda.
{% endhint %}

## 2. Antes de nada: prompting y metaprompting, en corto

Este estudio da por explicado en detalle el prompting y el metaprompting en la [Unidad 10](../unidades/u10-copiloto-datascience.md); aquí solo el resumen imprescindible para que el documento se entienda de forma autónoma.

**Prompting** es, sencillamente, la destreza de **pedir bien**. Un asistente sin contexto es como un compañero muy capaz que no conoce tu caso: hará algo razonable en abstracto, pero puede pasar por alto detalles clínicos que a ti te saltarían a la vista. Pedir bien significa darle, siempre, cuatro cosas:

* **Contexto** — qué son los datos, qué representa una fila, qué decisión clínica hay detrás.
* **Objetivo** — qué quieres conseguir exactamente (por ejemplo, "el modelo que mejor prediga `evento_cv`").
* **Métrica** — con qué número se juzga el éxito (en `evento_cv`, casi nunca la *accuracy*; mejor AUC, recall o PR-AUC, como vimos en la Unidad 5).
* **Criterio de éxito** — cuándo paras y qué harías con el resultado (un número mínimo a superar, un presupuesto de intentos, un test final que no se toca).

{% hint style="info" %}
**Concepto · El patrón "contexto + objetivo + métrica + criterio de éxito"**

Cualquier encargo de búsqueda o de ajuste a un asistente se vuelve fiable si responde a estas cuatro preguntas antes de ejecutar nada: **¿de qué datos partimos y qué representan? ¿qué queremos lograr? ¿con qué número lo medimos? ¿cuándo consideramos que ya está bien y hemos terminado?** Faltar cualquiera de las cuatro deja al asistente "improvisando" sobre un vacío que tú, como clínico, podrías haber rellenado en una frase.
{% endhint %}

**Metaprompting** es un paso más: en vez de confiar en que tu propio encargo esté bien planteado, le pides al asistente que **lo revise antes de ejecutar nada**. Es una red de seguridad especialmente valiosa para quien no programa, porque el asistente conoce los fallos típicos de un encargo de búsqueda de hiperparámetros —métrica ausente o mal elegida, ausencia de un test final reservado, presupuesto sin fijar— mejor que un principiante.

{% hint style="info" %}
**Concepto · Metaprompting**

Pedirle al asistente que **mejore tu propia instrucción** antes de ponerla en marcha: que señale qué falta (métrica, criterio de parada, riesgo de sobreajustar la validación) y te devuelva un encargo más completo. Es "usar la IA para pedirle mejor a la IA", y es el primer paso de cualquier bucle bien dirigido.
{% endhint %}

{% hint style="success" %}
**🤖 Prompt reutilizable · Metaprompting antes de lanzar la búsqueda**

```text
Antes de ejecutar nada, actúa como un experto en ajuste de hiperparámetros para
modelos clínicos y CRITICA este encargo mío:
1. ¿Qué me falta para que la búsqueda sea fiable (métrica, presupuesto de
   intentos, si reservo un test final que no toque la búsqueda)?
2. ¿La métrica que propongo (AUC / recall / accuracy...) encaja con la decisión
   clínica de priorizar a quién citar antes en prevención cardiovascular?
3. ¿Ves riesgo de que, si itero muchas veces contra la misma validación, acabe
   "sobreajustando la validación" en lugar de mejorar el modelo de verdad?

Luego reescríbeme el encargo ya corregido.

--- Mi encargo ---
Con 'pacientes.csv' (cohorte SINTÉTICA), busca los mejores hiperparámetros de un
Random Forest para predecir evento_cv.
```
{% endhint %}

## 3. Cuatro niveles de automatización del bucle de hiperparámetros

Aquí está el corazón del estudio. El mismo bucle conceptual —proponer una configuración, ejecutarla, medir, decidir— se puede montar con grados de automatización muy distintos, desde "pides el código en un chat y tú haces de mensajero" hasta "un agente lo hace casi todo solo". Los cuatro niveles no compiten entre sí: cada uno tiene su sitio según cuánto tiempo tengas, cuánta confianza quieras mantener sobre cada paso, y cuánta fricción de configuración estés dispuesto a asumir.

| Nivel | Quién cierra el bucle | ¿Hay que instalar algo? | Encaje para un sanitario |
| --- | --- | --- | --- |
| 0 · Chat manual | La persona (copia y pega) | No | El más didáctico; ves y entiendes cada vuelta |
| 1 · Asistente en Colab | El propio Colab (con tu aprobación) | No, o casi nada | El más práctico para el día a día |
| 2 · Agente local que itera | El agente, solo | Sí (herramienta + clave de acceso) | Demostración de a dónde va esto |
| 3 · Híbrido (agente + Colab) | El agente, ejecutando en Colab | Sí (una vez) | Idea de futuro; aún muy reciente |

### Nivel 0 · Pedir el código en un chat (cero instalación)

Es el punto de partida más simple y el más fácil de entender: te sientas delante de un asistente normal (Gemini, Claude, ChatGPT) y le pides una primera versión del código para buscar hiperparámetros. **Tú** lo ejecutas en Colab, copias el resultado (el AUC, la curva, el mensaje de error si lo hay) y se lo devuelves al asistente, que razona sobre esos números y te propone una versión mejorada. Repites. Tú eres el "cable" que conecta al asistente con la ejecución real: nada se automatiza, pero el patrón de auto-mejora ya está presente.

**✅ Fortalezas**

* Cero instalación: solo un chat y Colab, nada más
* Entiendes y controlas cada vuelta, porque la ves con tus propios ojos
* Funciona con cualquier asistente, incluso el más sencillo
* El mejor punto de partida para aprender el concepto

**⚠️ Límites y debilidades**

* Manual: copias y pegas código y resultados en cada vuelta
* Lento; no llega a muchas iteraciones seguidas
* El asistente "no ve" la ejecución real; depende de lo que le pegues tú
* Fácil perder el hilo del contexto entre vuelta y vuelta

{% hint style="success" %}
**🤖 Prompt reutilizable · Una vuelta del bucle manual (lo que le pegas al asistente en cada iteración)**

```text
Esta es la configuración de hiperparámetros que probé y este es el resultado
real que obtuve al ejecutarla en Colab sobre 'pacientes.csv' (cohorte SINTÉTICA):

[pego el código con la configuración]
[pego el resultado: AUC=0.83, recall=0.61, y la curva ROC]

Mi objetivo es predecir evento_cv (prevalencia ≈19%) para priorizar a quién citar
antes en prevención; me importa más no perder casos (recall) que la accuracy.
Analiza el resultado y propón UNA modificación concreta de hiperparámetros (por
ejemplo, profundidad, número de árboles). Explícame qué cambias y por qué
esperas que mejore el recall. No toques la partición train/test.
```
{% endhint %}

**Ejemplo trabajado (tres vueltas sobre `evento_cv`).** Así se vería en clase, anotando la métrica en cada iteración:

```
Vuelta 1  ·  El asistente propone: RandomForest con valores por defecto
          ·  Ejecuto en Colab  ->  AUC=0,80, recall=0,55 (mejorable)
          ·  Le pego el resultado.
Vuelta 2  ·  Analiza y propone: aumentar profundidad máxima y nº de árboles;
             "con pocos árboles el bosque no está capturando bien las
             interacciones entre glucemia, IMC y tabaquismo".
          ·  Ejecuto  ->  AUC=0,83, recall=0,63  ->  CONSERVO el cambio.
Vuelta 3  ·  Propone: bajar el umbral de decisión porque nos duele más el
             falso negativo que el falso positivo.
          ·  Ejecuto  ->  recall=0,71 (a costa de más falsos positivos, aceptable
             en este caso)  ->  CONSERVO.  Decido parar (presupuesto: 3 vueltas).
```

El valor didáctico de este nivel es enorme porque **cada decisión queda explícita y discutible**: puedes parar en cualquier vuelta, cuestionar la propuesta, o pedir una explicación clínica de por qué ese cambio debería ayudar. La desventaja es que no escala a treinta o cuarenta intentos; para eso está el siguiente nivel.

### Nivel 1 · Un asistente que itera dentro de Colab (con tu aprobación)

Aquí el bucle se automatiza **dentro del propio Colab**, sin instalar nada externo ni complicado. La idea es que, en lugar de que tú copies y pegues en cada vuelta, el propio entorno de Colab —con su asistente integrado— **recibe el objetivo, propone un plan, y genera y ejecuta las celdas** para cumplirlo, mostrándote el resultado real paso a paso. Tú apruebas el plan o pides ajustes, pero ya no haces de mensajero manual entre el chat y el notebook: la ejecución ocurre delante de ti, en el mismo sitio.

**✅ Fortalezas**

* No hay que instalar nada raro: todo ocurre en Colab
* El asistente integrado en Colab es gratuito y no exige configuración
* Ves el plan antes de que se ejecute, y puedes frenarlo o corregirlo
* Aprovecha directamente los recursos de cómputo de Colab

**⚠️ Límites y debilidades**

* Es menos controlable, paso a paso, que hacerlo tú mismo a mano
* Iteraciones largas consumen tiempo de sesión
* Sigue existiendo el riesgo de sobreajustar la validación si iteras sin límite
* Conviene revisar cada celda generada antes de fiarte de su resultado

{% hint style="success" %}
**🤖 Prompt reutilizable · Objetivo dado al asistente de Colab**

```text
He subido 'pacientes.csv' (cohorte SINTÉTICA, no son pacientes reales). Quiero
predecir evento_cv (prevalencia ≈19%) para priorizar a quién citar antes en
prevención cardiovascular. Prueba distintas configuraciones de hiperparámetros
de un Random Forest, evalúa cada una con validación cruzada, y busca la que
mejor recall obtenga sin hundir el AUC. Reserva un test final que no toques
durante la búsqueda, y muéstrame al final la mejor configuración, el AUC, el
recall y la curva de calibración.
```
{% endhint %}

_El asistente responde con un plan (cargar datos → preparar sin fuga → probar configuraciones → validar → elegir → evaluar en el test reservado) que apruebas; luego crea y ejecuta cada celda ante ti. Es "bucle asistido" sin salir de Colab ni instalar nada extra._

### Nivel 2 · Un agente local que itera solo (más potencia, más instalación)

Subimos un peldaño de autonomía. Existen herramientas —agentes de código que operan desde una terminal o un editor— capaces de **leer un fichero, ejecutar código de verdad, leer el resultado, reescribir el código y volver a ejecutar, sin que tengas que reinstruirlas en cada vuelta**. Aquí el bucle Proponer→Ejecutar→Observar está completamente cerrado por la propia herramienta: le das el fichero, la métrica y un presupuesto de intentos, y el agente entrena, mira el número real, decide si conserva o descarta, y repite solo.

La contrapartida es que esto **requiere instalar la herramienta** y una clave de acceso, y que la ejecución ocurre en tu propio ordenador, no en la nube de Colab. Es un salto de potencia real, pero también de fricción de configuración.

**✅ Fortalezas**

* El bucle queda totalmente cerrado: ejecuta, mide y decide sin intervención en cada paso
* Puede trabajar sobre varios ficheros de un proyecto, no solo uno
* Se ancla siempre en una métrica ejecutada de verdad, no en su propia impresión
* Es el patrón más cercano a cómo trabaja hoy la investigación puntera en este campo

**⚠️ Límites y debilidades**

* Exige instalar la herramienta y configurar el acceso, un paso técnico
* Ejecuta en tu máquina, sin la GPU gratuita de Colab
* Cada vuelta consume recursos; hay que fijar presupuesto de antemano
* Necesita un criterio de parada y una verificación bien definidos, o puede iterar sin rumbo

{% hint style="warning" %}
**⚠️ Aviso · La verificación separa un bucle controlado de una caja negra**

Da igual el nivel de automatización: hay que **anclar cada observación en una métrica real ejecutada**, no en la propia impresión del asistente sobre si "cree" que ha mejorado. Instruir explícitamente algo como *"tras cada cambio, ejecuta la validación cruzada y reporta el AUC exacto; no sigas sin ese número"* evita que el agente dé por buena una mejora que no es cierta. El criterio de parada —número de vueltas, mejora mínima, tiempo— también debe fijarse **antes** de empezar, no a mitad de camino.
{% endhint %}

**Ejemplo trabajado (patrón Karpathy aplicado a `evento_cv`).** Se prepara un único fichero que entrena, evalúa por validación cruzada e imprime la métrica, y se le dan al agente unas instrucciones de iteración muy simples:

```
Objetivo: maximizar el recall en validación cruzada, sin que el AUC baje de 0,80,
editando SOLO el fichero de entrenamiento del modelo de evento_cv.
Bucle:
 1. Propón UNA modificación de hiperparámetros.
 2. Ejecuta el entrenamiento y la validación cruzada; imprime el AUC y el recall.
 3. Si el recall sube y el AUC no baja de 0,80, conserva el cambio; si no, revierte.
 4. Repite. Presupuesto: 12 iteraciones. NO toques el test final reservado.
Reporta tras cada vuelta el número exacto; no te fíes de tu propia intuición.
```

```
[itera] prueba configuración -> AUC=0,82, recall=0,64 -> conserva
[itera] prueba configuración -> AUC=0,79, recall=0,70 -> descarta (AUC bajó del límite)
[itera] prueba configuración -> AUC=0,83, recall=0,69 -> conserva
... (12 vueltas) ...
> Mejor resultado: AUC=0,84, recall=0,73, alcanzado en la vuelta 9.
```

El bucle queda **totalmente cerrado**: el agente ejecuta, lee la métrica real y decide, sin que nadie haga de mensajero. La contrapartida —instalación local y ejecución en tu propia máquina— es justo lo que intenta resolver el siguiente nivel.

### Nivel 3 · Híbrido: agente potente ejecutando en Colab (idea de futuro)

Existe, además, una vía intermedia entre el Nivel 2 y el Nivel 1: un agente potente que en lugar de ejecutar en tu propio ordenador, **controla y ejecuta las celdas directamente en un notebook de Colab en la nube**. La idea es unir lo mejor de los dos mundos: la capacidad de iterar sola de un agente potente, con la ventaja de que el cómputo ocurre en la infraestructura de Colab y el notebook queda como un artefacto reproducible que cualquiera puede reabrir después.

Es tecnología muy reciente y todavía en maduración, así que aquí la dejamos solo como **idea de futuro**, sin entrar en el detalle de cómo se conecta ni de qué herramientas concretas hacen falta: lo importante para un sanitario no es la infraestructura, sino saber que este puente existe y que es la dirección hacia la que camina la automatización de estos bucles.

{% hint style="warning" %}
**⚠️ Aviso · Nivel 3, con cautela**

Al ser tan reciente, conviene tratarlo como una posibilidad emergente y no como algo ya asentado en la práctica clínica habitual. Como con cualquier nivel de automatización, si algún día lo usas: fija presupuesto, criterio de parada y reserva siempre un test final que la búsqueda no toque.
{% endhint %}

## 4. El patrón que no cambia: humano al mando, en cada nivel

Sube el nivel de automatización que sube, hay algo que **no cambia nunca**: el sanitario sigue **al mando del bucle**, no dentro de él picoteando código. En cada nivel, tu papel se concreta en cuatro gestos, y son los mismos cuatro que ya viste en la Unidad 10 aplicados aquí a la búsqueda de hiperparámetros:

* **Fijas el contexto y el objetivo** antes de arrancar cualquier nivel: qué decisión clínica hay detrás (priorizar citas de prevención), qué representa `evento_cv`, qué columnas no estarían disponibles en el momento de decidir.
* **Eliges la métrica**, y la eliges tú, no el asistente por defecto: en `evento_cv`, casi nunca la *accuracy* (con prevalencia ≈19 %, "decir que nadie tendrá el evento" acierta el 81 % y es inútil clínicamente); mejor AUC, recall o PR-AUC según qué error te duela más, exactamente como se explicó en la Unidad 5.
* **Fijas el criterio de éxito y el presupuesto**: cuántas vueltas, cuánta mejora mínima esperas, y sobre todo, que exista un **test final reservado** que la búsqueda nunca toque —el aviso de la Unidad 5 sobre "sobreajustar la validación" aplica exactamente igual aquí: si dejas que el bucle itere sin límite contra la misma validación, la cifra final te mentirá.
* **Validas el resultado con tu criterio clínico** al final de cada vuelta relevante: ¿la configuración que "gana" tiene sentido? ¿el gradiente de riesgo por tabaquismo o por HDL sigue siendo coherente tras el ajuste? ¿el AUC "demasiado bueno" no esconderá una fuga de datos?

{% hint style="warning" %}
**⚠️ Aviso · Un bucle acelera la búsqueda, no sustituye el criterio**

En ningún nivel de automatización el asistente sabe, por sí solo, qué error clínico duele más ni si una variable debería estar disponible en el momento de decidir. El bucle **itera**; el sanitario **gobierna**: define bien la métrica, evita la fuga de datos, fija el presupuesto y valida el resultado final sobre un test honesto que nunca participó en la búsqueda.
{% endhint %}

## 5. Qué nivel elegir, sin complicarse

Para cerrar, una guía práctica y honesta sobre cuándo usar cada nivel, sin sentir que hay que dominarlos todos:

* Si es la **primera vez** que ves este concepto, o quieres enseñarlo a otra persona, empieza siempre por el **Nivel 0**: es el más lento, pero el único en el que ves y entiendes cada decisión con total claridad.
* Para el **trabajo real del día a día** con Colab, el **Nivel 1** —el asistente integrado que planifica, ejecuta y te muestra el resultado— es probablemente el punto justo entre comodidad y control: no instalas nada, y sigues aprobando cada paso.
* El **Nivel 2** merece conocerse como **demostración de hacia dónde va esto**, sobre todo si tu entorno de trabajo ya usa herramientas de este tipo para otras tareas; no es imprescindible para sacar partido al resto del curso.
* El **Nivel 3** queda, por ahora, como **idea de futuro**: suficiente saber que existe y qué promete, sin necesidad de profundizar en su instalación.

{% hint style="success" %}
**💡 Idea clave**

El mensaje de fondo es el mismo de siempre en este curso, aplicado a un rincón muy concreto: "búsqueda de hiperparámetros" y "bucle de auto-mejora" son la misma idea —Proponer→Ejecutar→Observar— implementada con más o menos automatización. Tú eliges el nivel según tu contexto, tu tiempo y tus restricciones; lo que nunca delegas es el criterio con el que juzgas el resultado.
{% endhint %}

## Qué llevarte

* **Un bucle de auto-mejora es Proponer → Ejecutar → Observar**, con una métrica real y una regla de conservar o descartar. Aplicado a hiperparámetros, es la misma búsqueda que ya conocías (Unidad 5) pero con el asistente razonando sobre el histórico de intentos en lugar de probar a ciegas.
* **Prompting es pedir bien** (contexto + objetivo + métrica + criterio de éxito) y **metaprompting es pedirle al asistente que mejore tu propio encargo** antes de ejecutar nada. Ambos son la puerta de entrada a cualquier nivel de automatización.
* **Hay cuatro niveles**, de menos a más automatización: **chat manual** (tú de mensajero, cero instalación), **asistente en Colab** (itera con tu aprobación, sin instalar nada), **agente local** (itera solo, requiere instalación) e **híbrido agente+Colab** (idea de futuro, aún inmaduro). Ninguno sustituye a los anteriores.
* **El humano fija el contexto, la métrica y el criterio de éxito, y valida el resultado**, en todos los niveles. El bucle acelera la búsqueda; nunca sustituye el criterio clínico.
* **El riesgo de siempre sigue vigente**: si el bucle itera sin límite contra la misma validación, sobreajusta la validación (Unidad 5). Reserva siempre un test final que la búsqueda no toque.

***

Este estudio se apoya en dos piezas del curso que conviene tener a mano: la [**Unidad 5**](../unidades/u05-supervisados-ii.md), donde nace la búsqueda de hiperparámetros y el aviso sobre sobreajustar la validación, y la [**Unidad 10**](../unidades/u10-copiloto-datascience.md), donde se explica con detalle el prompting, el metaprompting y el papel del sanitario frente al agente. Si quieres ver el mismo patrón aplicado a un proyecto completo de principio a fin, el estudio hermano [**Copiloto de ciencia de datos**](copiloto-datascience.md) es el siguiente paso natural.
