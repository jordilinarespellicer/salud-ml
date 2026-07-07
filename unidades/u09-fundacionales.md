---
description: >-
  El giro definitivo de 2026: ya casi nadie entrena modelos desde cero. Somos
  usuarios de modelos ya entrenados. Los descargamos de Hugging Face con
  pipeline() o los llamamos por API (OpenRouter), y nuestro trabajo es elegir,
  integrar, evaluar y orquestar con criterio clínico.
---

# U9 · Modelos fundacionales — Hugging Face y APIs (OpenRouter)

En la unidad anterior cerramos con una idea que ahora se convierte en el eje de todo: **casi nadie entrena las grandes redes desde cero; las usamos ya entrenadas**. Aquí lo llevamos hasta sus últimas consecuencias.

Y de paso descubrimos que es la mejor noticia posible para un profesional sanitario que no programa. Porque si el trabajo ya no consiste en construir el modelo, sino en **elegirlo, integrarlo, evaluarlo y orquestarlo**, entonces lo que de verdad importa es exactamente lo que tú tienes y una máquina no: **criterio clínico**.

Durante décadas, "hacer machine learning" significó recopilar datos, diseñar una arquitectura y entrenarla con enorme esfuerzo y cómputo. Eso sigue existiendo, pero ha dejado de ser el camino habitual.

Hoy, un puñado de organizaciones con recursos colosales entrena **modelos fundacionales** —redes gigantescas, preentrenadas a una escala que ningún hospital ni universidad podría replicar— y los pone a disposición de todos. Nuestro papel ha cambiado de raíz: **de fabricantes a usuarios**. Y usar bien un modelo que otro entrenó es una habilidad distinta —y, en salud, más responsable— que entrenarlo.

{% hint style="success" %}
**💡 Idea clave**

**En 2026 somos, sobre todo, USUARIOS de modelos ya entrenados.** No los construimos: los **descargamos** de un repositorio como **Hugging Face** o los **llamamos por API** (por ejemplo, a través de OpenRouter). El valor profesional se ha desplazado a cuatro verbos: **elegir** el modelo adecuado, **integrarlo** en un flujo de trabajo, **evaluarlo** con honestidad clínica y **orquestar** varios cuando hace falta. Ninguno de esos cuatro exige saber programar de memoria; los cuatro exigen criterio.
{% endhint %}

### Objetivos de esta unidad

* Entender qué es un **modelo fundacional** y la diferencia radical entre **usar** un modelo y **entrenarlo**.
* Conocer **Hugging Face** —"el GitHub de los modelos"— y su función `pipeline()`, la forma más simple de poner un modelo a trabajar en **tres líneas**.
* Ver tres ejemplos clínicos reales del Hub: un clasificador de **radiografía de tórax**, un extractor de entidades biomédicas en **inglés** y un modelo fundacional biomédico-clínico en **español**.
* Saber llamar a los grandes modelos propietarios (Claude, GPT, Gemini, Qwen) desde un notebook con **una sola clave**, vía **OpenRouter**.
* Situar **Qwen** como ejemplo de potencia de primer nivel con **pesos abiertos** y coste casi nulo.
* Manejar las **advertencias clínicas** propias de este mundo: validación en tu población, **privacidad** de los datos y reproducibilidad.

## 9.1 Qué es un modelo fundacional (y por qué "usar" no es "entrenar")

Empecemos por el concepto que da nombre a la unidad. Un **modelo fundacional** es una red entrenada una sola vez, a gran escala, sobre una cantidad ingente de datos generales, de modo que aprende representaciones tan ricas y transferibles que **sirven de base ("fundación") para muchísimas tareas distintas**.

No se entrena para "detectar neumonía" ni para "resumir un informe": se entrena para "entender imágenes" o "entender lenguaje", y luego se **adapta** —o directamente se usa tal cual— para la tarea concreta.

{% hint style="info" %}
**Concepto · Modelo fundacional (*foundation model*)**

Modelo **preentrenado a gran escala** sobre datos amplios y variados, cuyo conocimiento general es **adaptable a un abanico enorme de tareas** posteriores. Los grandes modelos de lenguaje (los que hay detrás de un asistente conversacional) son el caso más conocido, pero también los hay de **imagen**, de **imagen + texto** (multimodales) o específicos de un dominio como el biomédico. La idea de U8 —el *transfer learning*— es su motor: se entrena una vez lo caro y general, y a partir de ahí se reaprovecha.
{% endhint %}

La distinción que de verdad tienes que interiorizar es **usar vs. entrenar**, porque cambia por completo lo que necesitas, lo que cuesta y de qué eres responsable.

* **Entrenar** un modelo (o incluso solo afinarlo mucho) exige **datos etiquetados**, **cómputo** (GPU), tiempo y conocimiento técnico. Entrenar uno *fundacional* desde cero está, sencillamente, fuera del alcance de cualquier organización sanitaria: hablamos de infraestructuras de coste astronómico.
* **Usar** un modelo ya entrenado es, en comparación, casi trivial: lo descargas o lo llamas, le pasas tu dato (una imagen, un texto) y recoges su respuesta. En **tres líneas**, como veremos enseguida.

{% hint style="success" %}
**💡 Idea clave**

La pregunta de un profesional de 2026 **casi nunca** es "¿cómo entreno un modelo para esto?", sino **"¿existe ya un modelo que resuelva esto, y es lo bastante bueno y fiable para mi caso?"**. Entrenar desde cero es la excepción rarísima; **buscar, elegir y evaluar** lo ya existente es el trabajo diario. Este cambio de mentalidad es, probablemente, lo más importante que te llevas de todo el curso.
{% endhint %}

## 9.2 Hugging Face: "el GitHub de los modelos"

Si los modelos ya están entrenados, ¿dónde están? La respuesta, en la inmensa mayoría de los casos, es **Hugging Face**.

Puedes pensar en él como **"el GitHub de los modelos"**: un enorme repositorio público —el **Hub**— donde la comunidad (empresas, universidades, laboratorios, particulares) **comparte** cientos de miles de modelos ya entrenados, junto con sus datos, su documentación y demos para probarlos. Buscas por tarea ("clasificación de imagen", "extracción de entidades"), por idioma o por dominio, lees su ficha, y te lo llevas.

{% hint style="info" %}
**Concepto · Hugging Face Hub**

Plataforma pública donde se **alojan y comparten** modelos ya entrenados (además de datasets y aplicaciones de demostración). Cada modelo tiene una **ficha** (*model card*) con su tarea, su licencia, cómo se entrenó y ejemplos de uso. Es el punto de partida natural cuando te preguntas "¿existirá ya un modelo para esto?": casi siempre, la respuesta está aquí.
{% endhint %}

Lo verdaderamente cómodo no es solo el catálogo, sino la **librería `transformers`** y su función estrella, `pipeline()`. Una *pipeline* empaqueta todo el proceso —descargar el modelo, preparar la entrada, ejecutarlo, ordenar la salida— detrás de una sola llamada.

El resultado es que **usar un modelo del Hub cabe, literalmente, en tres líneas**: importar, crear la *pipeline* con el nombre del modelo, y pasarle el dato.

{% hint style="info" %}
**Concepto · `pipeline()` de `transformers`**

Función de alto nivel de la librería `transformers` que, dado el **nombre de una tarea** (por ejemplo `"image-classification"`) y opcionalmente el **nombre de un modelo** del Hub, devuelve un objeto listo para usar: le das la entrada y te da la salida, encargándose por dentro de todos los pasos intermedios. Es la forma **más simple** de poner a trabajar un modelo preentrenado.
{% endhint %}

{% hint style="success" %}
**💡 Idea clave**

Que usar un modelo del Hub sean tres líneas **no lo convierte en apto para la clínica**. La facilidad de `pipeline()` es una invitación a probar, no un certificado de validez. Todo lo que aprendiste en U3 (métricas, calibración) y en U8 (validación externa, atajos espurios) sigue mandando: la parte fácil es ejecutarlo; la difícil —y la tuya— es **decidir si ese modelo sirve para tus pacientes**.
{% endhint %}

## 9.3 Tres modelos clínicos del Hub, en tres líneas cada uno

La mejor manera de entender esto es verlo. Vamos a "usar" tres modelos fundacionales ya entrenados, de tres tipos distintos, sin entrenar absolutamente nada.

Recuerda el reparto de papeles del curso: **tú decides qué modelo tiene sentido y qué preguntar; el código lo escribe el asistente**. Lo que sigue son fragmentos ilustrativos y comentados, para que reconozcas el patrón —no para memorizarlos—.

### Imagen: neumonía en radiografía de tórax (un ViT ya afinado)

En el Hub hay un modelo, `dima806/chest_xray_pneumonia_detection`, que es un **Vision Transformer (ViT)** —justo la arquitectura que vimos en U8— **ya afinado** para clasificar radiografías de tórax en *neumonía* / *normal*. No lo entrenamos: lo usamos.

```python
from transformers import pipeline

# El asistente elige un modelo ya afinado del Hub: un ViT para neumonía en Rx de tórax
clasificador = pipeline("image-classification",
                        model="dima806/chest_xray_pneumonia_detection")

# Le pasamos una radiografía (imagen PÚBLICA de ejemplo, NUNCA de un paciente real)
print(clasificador("radiografia_ejemplo.png"))
# → p. ej. [{'label': 'PNEUMONIA', 'score': 0.97}, {'label': 'NORMAL', 'score': 0.03}]
```

Tres líneas de trabajo y ya tenemos un clasificador de imagen médica respondiendo. Esto conecta de forma directa con U8: aquí no estamos montando la CNN ni el ViT, **estamos consumiendo uno que alguien ya entrenó y publicó**. Es el flujo por defecto de 2026.

{% hint style="warning" %}
**⚠️ Aviso: que responda no significa que valga para tus pacientes**

Un modelo del Hub puede haberse entrenado con radiografías de **otra población, otro equipo y otro protocolo** que los tuyos. Puede estar sesgado, mal calibrado o haber aprendido un **atajo espurio** (U8: aprender el aparato, no la enfermedad). Antes de fiarte, exige lo de siempre: ¿con qué datos se entrenó?, ¿está **validado en tu entorno**?, ¿en qué se fija? Un modelo público es un punto de partida para investigar, **no** una herramienta clínica lista para usar.
{% endhint %}

### Texto biomédico en inglés: extraer enfermedades, síntomas y fármacos

El segundo ejemplo es de texto. `d4data/biomedical-ner-all` es un modelo entrenado para **reconocer entidades biomédicas** en inglés —enfermedades, síntomas, fármacos— dentro de un texto libre. La tarea se llama `token-classification`.

```python
from transformers import pipeline

# NER biomédico (en inglés): detecta enfermedades, síntomas, fármacos... en texto libre
ner = pipeline("token-classification",
               model="d4data/biomedical-ner-all",
               aggregation_strategy="simple")

texto = "Patient with type 2 diabetes and hypertension, started on metformin."
for entidad in ner(texto):
    print(entidad["word"], "→", entidad["entity_group"])
# → diabetes → Disease_disorder ; hypertension → Disease_disorder ; metformin → Medication
```

De nuevo: sin entrenar nada, un modelo convierte una frase clínica en información estructurada. Aquí conviene una aclaración clínica: este modelo trabaja en **inglés**. Para el público hispanohablante de este curso, eso nos lleva al tercer ejemplo.

### Texto clínico en español: un modelo fundacional que se adapta

Para el mundo clínico en **español** existe `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es`, un modelo **RoBERTa** preentrenado sobre texto **biomédico y clínico en español**.

Es un ejemplo estupendo de modelo fundacional "de base": no viene resuelto para una tarea concreta, sino que **rellena huecos** (predice la palabra que falta), lo cual revela cuánto lenguaje clínico en español ha aprendido. Es, literalmente, **materia prima** que luego se **afina** para una tarea (extracción de entidades, clasificación de notas…).

```python
from transformers import pipeline

# Modelo FUNDACIONAL biomédico-clínico en español (RoBERTa). De base "rellena huecos":
# es la base que después se AFINA para una tarea concreta (p. ej. NER en español).
base = pipeline("fill-mask",
                model="PlanTL-GOB-ES/roberta-base-biomedical-clinical-es")

base("El paciente presenta dolor <mask> de esfuerzo.")
# → el modelo propone términos clínicos plausibles para <mask> (p. ej. "torácico", "precordial"...)
```

Este ejemplo ilustra perfectamente el mensaje de la unidad: un modelo fundacional en español, **preentrenado por otros** (aquí, en el marco del Plan de Tecnologías del Lenguaje), listo para que tú lo **adaptes** a tu problema con relativamente pocos datos, en lugar de entrenar desde cero un modelo del español clínico —algo impensable para un equipo asistencial—.

{% hint style="info" %}
**🏥 En la clínica · Un puente, no un curso de PLN**

Usamos este modelo en español como **ejemplo de "usar un modelo fundacional"** y como **puente** hacia el trabajo con texto clínico, no para enseñar el procesamiento del lenguaje natural, que tiene su propio espacio y su propia profundidad. Aquí nos basta con la idea grande: **existen modelos del lenguaje clínico en español, ya entrenados, que puedes tomar del Hub y adaptar**. El "cómo" fino del texto —tokenización, tipos de tarea, evaluación lingüística— pertenece a ese otro terreno.
{% endhint %}

Con estos tres ejemplos ya tenemos la primera vía completa: **descargar modelos de pesos abiertos y usarlos localmente** (en tu notebook de Colab) con `transformers`. Toca ver la segunda vía.

## 9.4 La otra vía: llamar a los grandes modelos por API (OpenRouter)

No todos los modelos potentes están para descargar. Los más grandes y capaces —los que hay detrás de los asistentes conversacionales de referencia: **Claude** (Anthropic), **GPT** (OpenAI), **Gemini** (Google)— son en buena medida **propietarios**: no se bajan, se **usan a distancia** a través de una **API**. Le mandas tu petición por internet a los servidores del proveedor, el modelo la procesa allí y te devuelve la respuesta. Tú no alojas nada.

El inconveniente clásico de este mundo es la fragmentación: cada proveedor tiene su cuenta, su clave, su facturación y su forma de llamar. Gestionar cinco servicios distintos para probar cinco modelos es un engorro. La solución más simple y barata para un notebook es un **agregador**, y el más cómodo es **OpenRouter**.

{% hint style="info" %}
**Concepto · OpenRouter (un endpoint para muchos modelos)**

Servicio que ofrece **un único punto de acceso** a **400+ modelos** de múltiples proveedores (Claude, GPT, Gemini, Qwen y muchos más) con **una sola clave** y **una sola factura**. Su gran ventaja práctica: es **compatible con la API de OpenAI**, así que puedes usar el mismo código de siempre y **solo cambiar la `base_url`** para apuntar a OpenRouter. Tiene modelos **gratuitos** y una capa de pago con un **pequeño recargo (~5,5%)** sobre el precio del proveedor.
{% endhint %}

La consecuencia es preciosa por lo económica que resulta en código: si ya sabes (o si tu asistente sabe) llamar a la API de OpenAI, **ya sabes usar 400+ modelos**. Solo cambias dos cosas —la dirección del servicio y la clave— y el resto es idéntico.

```python
from openai import OpenAI

# Mismo SDK de OpenAI; SOLO cambiamos la base_url (y la clave) para hablar con OpenRouter
cliente = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,   # nunca la escribas a mano en el notebook: usa un "secreto" de Colab
)

# Con UNA sola clave elegimos entre 400+ modelos (Claude, GPT, Gemini, Qwen...).
# Muchos tienen una variante gratuita (sufijo ":free"); las de pago llevan un recargo (~5,5%).
respuesta = cliente.chat.completions.create(
    model="qwen/...:free",   # copia el identificador exacto del catálogo de OpenRouter
    messages=[
        {"role": "system", "content": "Eres un asistente de análisis de datos clínicos."},
        {"role": "user",   "content": "Resume en 3 puntos los factores de riesgo que se "
                                       "aprecian en este resumen estadístico: ..."},
    ],
)
print(respuesta.choices[0].message.content)
```

Con este patrón, "llamar a un modelo potente desde un notebook" deja de ser un proyecto de infraestructura y se convierte en **una llamada más**. Es, con diferencia, la vía **más simple y barata** de tener a los grandes modelos trabajando para ti sin gestionar cinco cuentas.

{% hint style="info" %}
**🏥 En la clínica · Un uso realista de la API**

En el curso la usamos para tareas de **apoyo** sobre datos **sintéticos**: pedirle a un modelo que redacte el resumen en lenguaje natural de una tabla de resultados, que proponga hipótesis a explorar en un EDA, o que explique un hallazgo. Es un **copiloto** de análisis (lo desarrollamos en U10), no un dispositivo diagnóstico. Y con una regla innegociable que veremos en 9.6: **por esa API no viajan datos reales de pacientes.**
{% endhint %}

## 9.5 Qwen: potencia de primer nivel con pesos abiertos

Merece la pena detenerse en **Qwen**, la familia de modelos de **Alibaba**, porque encarna a la perfección hacia dónde va todo esto.

Qwen es un ejemplo de modelo **muy accesible** y, sobre todo, de **pesos abiertos**: buena parte de la familia se publica bajo licencia **Apache 2.0**, lo que significa que puedes **descargarla y usarla con enorme libertad**, incluso para fines comerciales. La familia es amplísima: **desde variantes de menos de 1.000 millones de parámetros** —que corren en hardware modesto— **hasta modelos grandes** de gran capacidad. (El modelo frontera, el de gama más alta tipo "Max", sí es cerrado; pero por debajo hay una escalera enorme de opciones abiertas.)

{% hint style="success" %}
**💡 Idea clave**

Qwen ilustra un cambio de fondo: **hoy hay potencia de primer nivel a un coste casi nulo**. Modelos abiertos y muy capaces que puedes descargar del Hub o llamar (gratis o casi) por OpenRouter. Para un profesional sanitario esto derriba la última barrera: el acceso a modelos punteros **ya no es cuestión de presupuesto**, sino de **criterio para elegirlos y usarlos bien**. La ventaja competitiva no está en tener el modelo; está en **saber qué pedirle y cómo evaluar su respuesta**.
{% endhint %}

Fíjate en que Qwen ejemplifica, él solo, las **dos vías** de la unidad: sus variantes abiertas puedes **descargarlas** de Hugging Face y ejecutarlas con `transformers`; y a la vez puedes **llamarlas por API** vía OpenRouter sin instalar nada. Abierto y accesible: la combinación que define este momento.

## 9.6 Dos vías, una decisión: ¿descargar de Hugging Face o llamar por API?

Ya tienes las dos formas de "usar un modelo ya entrenado". No compiten: se eligen según el caso. Vamos a tratarlas como las dos "técnicas" de la unidad, con sus fortalezas, sus límites y su encaje clínico.

### Vía A · Modelos de pesos abiertos con Hugging Face (en local)

Descargas el modelo del Hub y lo ejecutas en tu entorno (tu notebook de Colab, un servidor del hospital).

**✅ Fortalezas**

* **Control y privacidad**: el dato **no sale de tu entorno**; el modelo viene a los datos, no al revés. Es la vía natural cuando la información es sensible.
* **Reproducibilidad**: fijas la **versión** del modelo y obtendrás siempre lo mismo; nadie te lo cambia por detrás.
* **Coste marginal casi nulo** una vez descargado, y sin depender de la conexión ni de un proveedor externo.
* Enorme **catálogo** especializado, incluidos modelos clínicos y en español.

**⚠️ Debilidades y límites**

* Necesitas **cómputo propio** (a menudo **GPU**) para los modelos grandes.
* Requiere algo más de montaje que una simple llamada, y **mantener** las versiones.
* Los modelos abiertos de tamaño manejable **no siempre igualan** a los mayores modelos propietarios en las tareas más exigentes de lenguaje.

**Campo de aplicación clínica.** La vía por defecto cuando **la privacidad manda** o cuando existe un modelo **especializado** que encaja con tu tarea: clasificar imagen médica con un modelo del Hub, extraer entidades de texto clínico, partir de un modelo biomédico en español y afinarlo. Es también la vía coherente con desplegar **dentro** de la infraestructura sanitaria.

### Vía B · Modelos propietarios por API (OpenRouter)

Llamas al modelo a distancia; el cómputo lo pone el proveedor.

**✅ Fortalezas**

* Acceso inmediato a los **modelos más capaces** (Claude, GPT, Gemini) sin hardware propio.
* **Simplicidad máxima**: una clave, una llamada; con OpenRouter, **400+ modelos** intercambiables cambiando una línea.
* **Escala sin gestión**: no mantienes ni actualizas nada; siempre tienes la última versión disponible.
* Ideal para **prototipar** y para tareas de lenguaje general (resumir, explicar, redactar).

**⚠️ Debilidades y límites**

* **El dato sale de tu entorno**: viaja a un tercero. Esto es un **problema de privacidad** de primer orden en salud (ver 9.7... y tómatelo en serio).
* **Coste por uso** y **dependencia** de un proveedor y de su conexión.
* **Menos reproducibilidad**: el proveedor puede actualizar el modelo y cambiar sutilmente las respuestas.

**Campo de aplicación clínica.** Perfecta para el **copiloto de análisis** sobre datos **sintéticos o agregados/anonimizados**: generar el resumen en lenguaje natural de un análisis, proponer hipótesis, documentar un notebook, explicar un resultado. **No** para procesar datos identificables de pacientes salvo que exista un marco legal y técnico que lo garantice.

{% hint style="success" %}
**💡 Idea clave**

Las dos vías no compiten, se complementan según qué mande en cada caso: si **manda la privacidad** o existe un modelo **especializado**, vas a Hugging Face en local; si **manda la capacidad** de lenguaje o quieres **prototipar rápido** sobre datos sintéticos, vas a la API por OpenRouter. Decidir bien entre ambas es, en sí mismo, parte de tu criterio clínico.
{% endhint %}

## 9.7 Advertencias clínicas: usar un modelo ajeno es una responsabilidad

Usar es fácil; usar con responsabilidad es la asignatura de verdad. Tres avisos, por orden de gravedad.

{% hint style="danger" %}
**⚠️ Privacidad: por una API pública NO viajan datos reales de pacientes**

Cuando llamas a un modelo por API, **tu texto o tu imagen salen de tu ordenador** y llegan a los servidores de un tercero. Enviar por ahí una nota clínica, una imagen o cualquier dato identificable de un paciente es, sin las garantías legales y contractuales adecuadas, una **fuga de datos** y una posible infracción grave de la normativa de protección de datos. Regla práctica para este curso y para tu trabajo: a una API pública, **solo datos sintéticos, anonimizados o agregados**. Si necesitas procesar datos reales sensibles con un modelo, la vía es **local** (Vía A) o un despliegue con las debidas garantías dentro de la organización.
{% endhint %}

{% hint style="warning" %}
**⚠️ Validación: un modelo del Hub puede no valer para tu población**

Que un modelo esté publicado y "funcione" no dice **con quién** funciona. Se entrenó con unos datos, de una población, un equipo y un contexto que **puede no parecerse al tuyo**. Toda la lección de U8 aplica igual a los modelos ajenos: exige saber cómo se entrenó, busca **validación en tu entorno** y mira el rendimiento **por subgrupos**. "Funciona en el paper" ≠ "funciona en mi hospital", tanto si el modelo lo entrenaste tú como si te lo bajaste del Hub.
{% endhint %}

{% hint style="warning" %}
**⚠️ Reproducibilidad y versiones: fija lo que usas**

Un modelo no es una entidad inmutable. En el Hub, los modelos se **versionan** y evolucionan; en una API, el proveedor puede **actualizarlos** sin avisar, y una misma pregunta puede dar respuestas distintas con el tiempo. Para cualquier uso serio —y no digamos en investigación o clínica— **anota qué modelo y qué versión exacta usaste**, y prefiere las vías que te permitan **fijarla**. La reproducibilidad no es un lujo académico: es parte de la trazabilidad que la medicina exige.
{% endhint %}

{% hint style="success" %}
**💡 Idea clave**

El resumen de las tres advertencias cabe en una frase: **la facilidad de usar un modelo ajeno no reduce tu responsabilidad; la aumenta.** Como es tan fácil, la tentación de saltarse la privacidad, la validación y la trazabilidad es mayor. El criterio clínico —tuyo— es justo lo que impide que "tres líneas de código" se conviertan en un problema para un paciente.
{% endhint %}

## 9.8 ¿Qué vía elegir? Un mapa rápido

Reuniéndolo todo, la decisión se vuelve sencilla si te haces las preguntas en orden.

| Situación / necesidad | Vía recomendada | Por qué |
| --------------------- | --------------- | ------- |
| Dato **sensible** de paciente, no puede salir | **Hugging Face local** (Vía A) | El modelo va a los datos; nada sale de tu entorno |
| Existe un **modelo especializado** (imagen médica, texto clínico) | **Hugging Face** (`pipeline()`) | Modelo ya afinado para tu tarea, en tres líneas |
| Necesitas **la máxima capacidad** de lenguaje ya | **API vía OpenRouter** | Acceso inmediato a Claude/GPT/Gemini sin hardware |
| **Prototipar** rápido sobre datos sintéticos/agregados | **API vía OpenRouter** | Una clave, una llamada, 400+ modelos |
| **Reproducibilidad** estricta y control de versiones | **Hugging Face local** (Vía A) | Fijas la versión; nadie la cambia por detrás |
| Quieres **potencia abierta y barata** | **Qwen** (abierto: local o API) | Pesos abiertos Apache 2.0, de <1B a modelos grandes |
| **Entrenar desde cero** un fundacional | Prácticamente **nunca** | Fuera del alcance de una organización sanitaria |

La última fila es la moraleja de la unidad: entrenar un fundacional desde cero no está en tu horizonte, y no pasa nada. Tu trabajo es **elegir bien** entre lo que ya existe.

## 9.9 Práctica en Colab

{% hint style="success" %}
**🔬 Práctica en Colab** — `U09a_Fundacionales_HF.ipynb` · [Abrir en Colab](https://colab.research.google.com/drive/1MgvsLL_6QYwjP1fPM7JWGEJKpfJWBCY8)

**Usar modelos ya entrenados de Hugging Face, sin entrenar nada.** El notebook recorre las tres piezas de la sección 9.3 con `pipeline()`: **(1)** clasificar una **radiografía de tórax** de ejemplo (imagen **pública**, nunca de un paciente real) con el ViT `dima806/chest_xray_pneumonia_detection`; **(2)** extraer entidades biomédicas en inglés con `d4data/biomedical-ner-all`; y **(3)** asomarse al modelo fundacional en **español** `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es`, aplicándolo a frases tomadas de `notas_clinicas.csv` (**dataset sintético**; su primera celda lo genera, sin descargar nada). El objetivo no es la métrica, sino **sentir lo fácil que es "consumir" un modelo** y lo cuidadoso que hay que ser antes de fiarse.
{% endhint %}

{% hint style="success" %}
**🔬 Práctica en Colab** — `U09b_APIs_OpenRouter.ipynb` · [Abrir en Colab](https://colab.research.google.com/drive/1MU72hahjS96Nf1I3RfXOsqJkHrSJq_OI)

**Llamar a los grandes modelos por API con una sola clave.** El notebook conecta con **OpenRouter** (cambiando solo la `base_url` del SDK de OpenAI) y usa un modelo —por ejemplo una variante **gratuita** de **Qwen**— para **analizar un dataset sintético**: le pasamos el resumen estadístico de `pacientes.csv` (**sintético**; se genera en la primera celda) y le pedimos que **redacte los factores de riesgo en lenguaje natural** y proponga hipótesis. La clave se guarda como **secreto** de Colab, no en el código. Regla de oro recordada en el propio notebook: **por la API, solo datos sintéticos**.
{% endhint %}

**🤖 Prompt para el asistente · Usar un modelo del Hub en tres líneas**

```
En español y por celdas, con la librería transformers de Hugging Face:
1. Explícame en dos frases qué es una "pipeline" y por qué me deja usar un modelo
   ya entrenado sin entrenar nada.
2. Crea una pipeline de "image-classification" con el modelo
   dima806/chest_xray_pneumonia_detection y clasifícame una radiografía de tórax
   de EJEMPLO (una imagen pública; recuérdame por qué no debo usar la de un paciente).
3. Crea una pipeline de "token-classification" con d4data/biomedical-ner-all y
   extráeme las entidades de una frase clínica en inglés.
4. Al final, enumérame qué debería comprobar antes de fiarme de estos modelos en mi
   hospital (población, validación externa, sesgos, versión).
```

*Fíjate en el punto 4: no le pides solo que ejecute, le pides el **cuestionario clínico**. Ese es el criterio que distingue a un profesional de un usuario ingenuo de la IA.*

## 9.10 Qué llevarte

* **Somos usuarios, no fabricantes.** En 2026 ya casi nadie entrena modelos desde cero: los **descargamos** (Hugging Face) o los **llamamos por API** (OpenRouter). El trabajo es **elegir, integrar, evaluar y orquestar** —y eso es criterio, no programación—.
* **Modelo fundacional** = red preentrenada a gran escala, **adaptable** a muchas tareas. **Usar** uno es trivial; **entrenar** uno fundacional está fuera del alcance de cualquier hospital.
* **Hugging Face** es "el GitHub de los modelos": el **Hub** más la función **`pipeline()`** ponen un modelo a trabajar en **tres líneas**. Lo probamos con un **ViT** de neumonía, un **NER** biomédico en inglés y un **RoBERTa** biomédico-clínico en **español**.
* **OpenRouter** da acceso a **400+ modelos** (Claude, GPT, Gemini, **Qwen**) con **una clave**, siendo compatible con la API de OpenAI: **solo cambias la `base_url`**. Tiene opción **gratuita** y una de pago con un pequeño recargo (~5,5%).
* **Qwen** demuestra que hay **potencia de primer nivel con pesos abiertos** (Apache 2.0, de <1B a modelos grandes) a coste casi nulo: el acceso ya no es cuestión de presupuesto.
* **Responsabilidad, elevada:** a una API pública, **solo datos sintéticos o anonimizados**; valida el modelo en **tu población**; y **fija y anota la versión** por reproducibilidad. Usar es fácil; usar bien es tu trabajo.

Y aquí se cierra el círculo del curso. Ya no solo sabes qué familia de modelos encaja en cada problema clínico (U2–U8): ahora sabes también **dónde están los modelos ya hechos y cómo ponerlos a trabajar**. La última pieza es aprender a **dirigir** todo esto de forma fluida —a convertir a la IA en tu copiloto de ciencia de datos de principio a fin—, que es exactamente lo que veremos en la **U10**.
