---
description: >-
  Encuadre práctico del curso: qué sabe hacer la IA hoy en salud (y qué no),
  qué cambia con los modelos fundacionales, cuál es tu papel como profesional
  sanitario y por qué este curso es posible sin saber programar.
---

# U1 · ¿Qué está pasando con la IA (y qué implica en salud)?

Esta unidad es deliberadamente distinta a las demás: **corta, síncrona y sin práctica**.

No es una segunda introducción a la inteligencia artificial —esa ya la tenéis en el programa, a cargo de otro docente, con sus definiciones, la distinción entre IA fuerte y débil y el mapa de áreas del campo—, así que aquí no repetiremos nada de eso. Esta unidad es la **brújula antes del mapa**: vamos directos a tres preguntas prácticas.

Primera: **¿qué está funcionando de verdad, hoy, en salud?** Segunda: **¿qué no funciona (todavía, o quizá nunca)?** Y tercera: **¿qué implica todo esto para tu papel como profesional sanitario** —y, de paso, para el curso que empieza—?

La respuesta honesta a las tres cabe en una sesión breve, y es la que da sentido a todo lo que viene después.

### Objetivos de esta unidad

* Situar el momento actual: la era de los **modelos fundacionales** y el cambio de mentalidad **"de entrenar a usar"**.
* Un panorama honesto, **por tipo de dato**, de lo que la IA hace bien hoy en salud... y de lo que se le resiste.
* Un encuadre claro de tu papel: la IA como **copiloto que amplifica el criterio clínico**, no como sustituto.
* Los límites y trampas que conviene llevar en la cabeza desde el primer día (se profundizan en la U11).
* La promesa del curso: **mecanismos y criterio, sin necesidad de escribir código**.

## 1.1 El momento: de entrenar a usar

Durante décadas, aplicar IA a un problema concreto significaba un proyecto a medida: reunir datos propios, entrenar un modelo específico para esa tarea, validarlo y mantenerlo.

Ese enfoque sigue existiendo —y es, de hecho, el que aprenderemos a mirar con criterio en este curso—, pero en los últimos años se le ha sumado algo cualitativamente nuevo.

{% hint style="info" %}
**Concepto · Modelo fundacional**

Modelo de gran tamaño **preentrenado con cantidades enormes de datos generales** (texto, imagen, audio...) que después se adapta o se usa directamente para muchas tareas distintas.

En lugar de entrenar un modelo desde cero para cada problema, se parte de uno que "ya sabe mucho" y se le orienta. Los asistentes conversacionales que usamos a diario son su cara más visible.
{% endhint %}

Este cambio desplaza el centro de gravedad: pasamos **de entrenar a usar**. Para un profesional sanitario tiene tres consecuencias muy concretas:

* Muchas capacidades de IA ya no llegan como "un proyecto de datos", sino **integradas en herramientas** que se usan sin verlas: el dictado que transcribe, el borrador de informe, la priorización de una lista de trabajo.
* **Usar bien un modelo ya entrenado** —saber qué pedirle, qué esperar de él y dónde falla— se ha convertido en una competencia profesional en sí misma, tan importante como antes lo era entrenar.
* Y, a la vez, **entrenar modelos propios sigue teniendo todo el sentido** para problemas locales con datos tabulares (riesgo, demanda, reingresos): exactamente el terreno de este hilo de Machine Learning.

{% hint style="success" %}
**💡 Idea clave**

El cuello de botella ya no es "quién me programa esto", sino **quién tiene criterio para elegir la herramienta, validarla en su contexto y supervisarla**. Ese criterio es justo lo que un profesional sanitario puede —y debe— aportar. Este curso existe para dártelo.
{% endhint %}

## 1.2 Panorama honesto 2026: qué funciona hoy, por tipo de dato

Lo que sigue no es un catálogo de maravillas ni una lista de miedos. Es el patrón que aparece cuando se mira el campo con calma: la IA rinde en **tareas concretas, repetitivas y bien definidas**, con datos abundantes y una verdad de referencia razonablemente clara; y se atasca cuando la tarea exige integrar contexto, valores del paciente y responsabilidad.

Recorrámoslo por tipo de dato, que además será la estructura del propio curso.

### Imagen: el terreno más maduro

Es donde la IA lleva más años demostrando valor real. En **radiología**, sistemas que detectan y priorizan hallazgos en radiografía y TC, o que actúan como segunda lectura en programas de cribado.

En **dermatología**, clasificación de lesiones cutáneas a partir de fotografías. En **fondo de ojo**, cribado de retinopatía diabética que permite filtrar a gran escala quién necesita ser visto por un especialista. En **patología digital**, búsqueda y recuento de estructuras en preparaciones digitalizadas y priorización de las zonas sospechosas del porta.

Buena parte de estos sistemas están ya autorizados como producto sanitario en distintas jurisdicciones.

¿Por qué funciona aquí? Porque son **tareas perceptivas acotadas**: una imagen, una pregunta concreta, mucho dato histórico y una referencia relativamente objetiva contra la que evaluar.

Lo que estos sistemas **no** hacen es el diagnóstico integral del paciente: funcionan como detector y priorizador incansable, no como sustituto del especialista.

### Señal: ECG, monitorización y wearables

En **ECG**, detección de arritmias y anomalías del trazado, y priorización de registros en contextos de alto volumen. En **monitorización continua y wearables**, señales individualmente modestas —frecuencia cardiaca en reposo, actividad, sueño— pero valiosas por su continuidad: permiten ver tendencias que una consulta puntual jamás vería.

{% hint style="warning" %}
**⚠️ Aviso**

Aplicar estos detectores a población general de bajo riesgo **genera falsos positivos y consultas evitables**. El valor de una alerta depende críticamente de a quién se aplica; le pondremos números y nombre formal (**prevalencia**, **valor predictivo**) en la U3.
{% endhint %}

### Texto: la ola más transversal

Con los modelos fundacionales de lenguaje, el texto clínico ha pasado de ser el dato más difícil al más accesible: **resumir historias largas, estructurar informes, generar borradores de cartas y de informes, asistir en codificación, transcribir la consulta de forma ambiental**.

Es probablemente el cambio que antes tocará al mayor número de profesionales, porque ataca directamente la carga documental.

También aquí está el matiz más serio: un modelo de lenguaje produce texto **plausible aunque sea incorrecto**. Sin revisión humana no hay uso clínico serio, punto. (El módulo de PLN y modelos de lenguaje del programa entra a fondo en esto; nosotros lo tocaremos desde el ML en la U9.)

### Tabular: el territorio de este curso

Es el menos vistoso y uno de los de mayor valor real: **scores de riesgo aprendidos del histórico, apoyo al triaje, predicción de reingreso, de deterioro, de no presentación a citas, de ocupación y demanda de urgencias**.

Aquí es donde más modelos se entrenan "en casa", con datos del propio sistema de salud... y por eso es el terreno que **más criterio local exige**: es donde la trampa de "funcionaba en el paper" aparece con más frecuencia.

### El patrón, en dos listas

**✅ Dónde rinde hoy la IA en salud**

* **Cribado, triaje y priorización** de alto volumen: imagen, señal, listas de trabajo.
* **Segunda lectura**: detectar lo que el ojo cansado pasa por alto, sin sustituir la primera.
* **Borradores de texto clínico** y reducción de carga documental, siempre con revisión.
* **Predicción sobre datos tabulares propios**: riesgo, demanda, ocupación, reingreso.

**⚠️ Dónde se atasca (hoy)**

* El **diagnóstico integral**: integrar historia, exploración, contexto y valores del paciente en una decisión con responsabilidad.
* **Generalizar** a poblaciones, equipos y centros distintos de aquellos con cuyos datos se entrenó.
* Tareas sin **verdad de referencia clara**, o con datos escasos y ruidosos.
* Cualquier uso **sin supervisión** en el que un error plausible pueda pasar inadvertido.

{% hint style="success" %}
**💡 Idea clave**

El criterio para separar promesa de realidad cabe en una pregunta: **¿es una tarea concreta, repetible y evaluable, o exige el juicio integral sobre un paciente?** En la primera, la IA ya es (a menudo) excelente. En la segunda, hoy por hoy, es una herramienta en manos de quien decide.
{% endhint %}

Estos cuatro tipos de dato no son solo el panorama externo: son el **plan del curso**.

Trabajaremos con datos **sintéticos** —generados con código, sin ningún paciente real detrás—: una tabla de pacientes con factores de riesgo cardiovascular (`pacientes.csv`), una serie temporal de ingresos en urgencias (`urgencias_diarias.csv`), notas clínicas de texto (`notas_clinicas.csv`) y registros de un wearable (`wearable.csv`). El mundo real es el de esta unidad; los datos sintéticos serán nuestro **simulador de vuelo**.

## 1.3 Tu papel: copiloto humano al mando

La palabra que mejor describe la relación sana con estos sistemas es **copiloto**: la IA amplifica tu capacidad —ve más volumen, no se cansa, redacta en segundos, no olvida un patrón— mientras **el criterio y la responsabilidad siguen siendo tuyos**.

Deontológica y legalmente, la decisión firmada es de un profesional; los sistemas se conciben y autorizan como apoyo, no como sustituto.

Y el copiloto necesita del piloto exactamente las dos cosas que la IA no tiene:

* **La pregunta correcta.** Qué problema de tu servicio merece resolverse, qué variables tienen sentido clínico, qué resultado sería sospechoso, qué error sería inaceptable. Ningún modelo aporta eso; lo aporta quien conoce la clínica.
* **La validación.** ¿Es cierto? ¿Es seguro? ¿Aplica a *este* paciente, en *este* centro? La salida de una IA se parece más a un resultado de laboratorio que a un veredicto: información valiosa que se **integra** en el juicio clínico. Nadie trata un potasio sin mirar al paciente.

{% hint style="warning" %}
**⚠️ Aviso**

El riesgo silencioso del copiloto es el **sesgo de automatización**: aceptar la sugerencia de la máquina sin cuestionarla, precisamente porque *suele* acertar.

La supervisión humana tiene que ser **real y significativa**, no un clic de trámite. Si un sistema acierta, pongamos, el **95 %** de las veces, tu valor profesional está en el **5 %** restante... y en saber reconocer cuándo estás en él.
{% endhint %}

## 1.4 Límites reales y trampas frecuentes

Cuatro trampas que conviene llevar puestas desde hoy. No para desconfiar de todo, sino para preguntar bien:

* **Sesgo.** Un modelo aprende del histórico, y el histórico contiene desigualdades: grupos infrarrepresentados, acceso desigual, registro desigual. Un sistema puede rendir bien "de media" y mal justo en el subgrupo que más te importa. La pregunta: *¿con qué población se entrenó y cómo rinde por subgrupos?*
* **"Funciona en el paper ≠ funciona en mi hospital".** Es la trampa más frecuente. Otra población, otro escáner, otra prevalencia, otras rutinas de registro... y el rendimiento cae. A esto se le llama falta de **validación externa**, y es la razón por la que muchos modelos prometedores nunca sobreviven al contacto con un centro distinto. La pregunta: *¿se ha evaluado en un contexto como el mío, con datos que el modelo nunca vio?*
* **Datos sensibles.** Los datos de salud tienen el máximo nivel de protección. Eso condiciona qué herramientas pueden usarse, dónde se procesan los datos y con qué garantías. (El módulo de aspectos legales del programa lo trata a fondo; aquí nos basta la regla de oro del aviso de abajo.)
* **Cajas negras.** Muchos modelos potentes no permiten ver *por qué* dicen lo que dicen. En clínica eso importa: necesitamos poder detectar el fallo absurdo, entender a qué se agarra el modelo y saber cuándo fiarnos. De ahí el interés creciente por la **explicabilidad**, que también veremos con herramientas concretas.

{% hint style="danger" %}
**⚠️ Aviso**

Dos líneas rojas desde el primer día: **nunca introduzcas datos identificables de pacientes reales en herramientas no autorizadas por tu organización** (un chatbot público incluido), y **nunca uses con pacientes una herramienta no validada ni autorizada para ese uso**.

Todo lo que hagamos en este curso ocurre sobre datos sintéticos, precisamente por esto.
{% endhint %}

Nada de esto es un apéndice ético para el final: es ingeniería de la buena y práctica clínica de la buena. En la **U11** le dedicaremos una unidad entera —sesgo, validación, privacidad y explicabilidad— cuando ya tengáis los mecanismos para entenderla a fondo.

## 1.5 La promesa del curso: mecanismos y criterio; el código lo escribe la IA

Queda la pregunta que muchos os estaréis haciendo: *"todo esto está muy bien, pero yo no sé programar"*. Y aquí está la buena noticia que hace posible este curso: **en 2026 no hace falta**.

La barrera que durante años reservó el Machine Learning a perfiles técnicos —escribir código— la ha derribado la propia IA. Los asistentes actuales escriben el código de un análisis o de un modelo a partir de una petición bien formulada en lenguaje natural.

Lo que **no** pueden poner por ti son las dos cosas que trabajaremos unidad a unidad:

* **Los mecanismos**: qué hace de verdad un modelo cuando "aprende", qué significa que "funcione", dónde y cómo se rompe.
* **El criterio**: qué pedir, qué comprobar, qué preguntas incómodas hacer antes de fiarte de un resultado.

Al terminar, deberías ser capaz de leer con otros ojos la próxima promesa comercial o el próximo artículo "con IA" que llegue a tu servicio; de formular un problema de tu entorno en términos de ML; de construir un prototipo sobre datos sintéticos con un asistente al lado; y de saber qué falta entre ese prototipo y algo usable con pacientes.

{% hint style="success" %}
**💡 Idea clave**

No vas a aprender a programar: vas a aprender a **pensar en Machine Learning**. La diferencia entre un usuario deslumbrado y un profesional con criterio no está en el código —ese, en 2026, se pide—, sino en entender qué hay debajo y qué preguntas hacer.
{% endhint %}

## 1.6 Dónde encaja esta parte en el programa

Este hilo de **Machine Learning** convive con otros módulos del curso que no vamos a duplicar: la **introducción general a la IA** (que ya encuadra definiciones y áreas), un módulo de **PLN y modelos de lenguaje**, otro de **radiómica e imagen avanzada**, y los **aspectos legales y éticos** tratados desde el derecho.

Lo nuestro son los **mecanismos comunes** que hay debajo de todos ellos: cómo aprende una máquina a partir de datos y cómo se evalúa si lo hace bien. Cuando en el módulo de imagen os hablen de un clasificador, o en el legal de un "sistema de alto riesgo", los mecanismos serán los de aquí.

{% hint style="info" %}
**Concepto · El recorrido del hilo de Machine Learning**

De la **U2** a la **U8** construimos el edificio (datos y EDA → métricas → modelos supervisados → no supervisado → series temporales → redes, imagen y señal); la **U9** y la **U10** lo conectan con los modelos fundacionales y con el trabajo asistido por IA; y la **U11** cierra con ética, sesgo y validación, ya con fundamento.
{% endhint %}

## Qué llevarte

* La IA en salud **ya funciona**, pero donde funciona es en **tareas concretas, repetibles y evaluables** —cribar, priorizar, redactar borradores, predecir sobre datos propios—, con datos suficientes y supervisión humana. El "diagnosticador universal" no existe (hoy).
* Vivimos el cambio **"de entrenar a usar"**: los modelos fundacionales convierten el **criterio** —no el código— en la habilidad escasa.
* Tu papel es de **copiloto humano al mando**: aportas la pregunta correcta y validas la respuesta. La responsabilidad no se delega, y el sesgo de automatización es el enemigo silencioso.
* Desconfía por sistema de "funcionó en el paper": entre la promesa y tu hospital se interponen el **sesgo, la validación externa, los datos sensibles y las cajas negras** (lo desarrollamos en la U11).
* Este curso te dará **mecanismos y criterio**; el código lo escribirá la IA contigo. Por eso es posible aunque nunca hayas programado.

Con la brújula puesta, toca abrir el mapa: en la **U2** empezamos a construir en serio —el vocabulario del ML, el método de trabajo 2026 con asistentes de IA y el primer contacto con nuestros datos sintéticos de pacientes.
