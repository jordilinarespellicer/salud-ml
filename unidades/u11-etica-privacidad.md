---
description: >-
  La lente transversal del curso: qué debe exigir un profesional sanitario a
  cualquier sistema de IA antes de fiarse — sesgo y equidad, validación clínica
  real, privacidad de los datos, encuadre regulatorio y supervisión humana.
---

# U11 · Ética, sesgo, validación y privacidad

{% hint style="warning" %}
**🔒 Unidad en preparación (todavía no disponible).** Esta unidad forma parte del temario, pero **aún no está cerrada**: su contenido puede cambiar. Por ahora, el curso publicado llega hasta la **U3**; las siguientes se irán liberando en las próximas semanas.
{% endhint %}


Esta unidad es distinta de las demás, y deliberadamente **breve**. No presenta ninguna técnica nueva: es la **lente transversal** con la que hay que mirar todo lo anterior.

A lo largo del curso has aprendido a construir y evaluar modelos con honestidad técnica; aquí cerramos con la otra mitad del criterio: **qué debe exigir un profesional sanitario a cualquier sistema de IA** —lo haya construido él, su hospital o un proveedor— antes de dejar que toque una decisión clínica.

Cuatro preguntas ordenan la unidad: **¿es justo con todos mis pacientes?**, **¿está validado de verdad (y lo seguirá estando)?**, **¿respeta sus datos?** y **¿quién decide y quién responde?**

{% hint style="success" %}
**💡 Idea clave**

La ética de la IA clínica no es un añadido al final del proyecto: es **parte del criterio clínico**. La pregunta madura no es "¿qué AUC tiene este modelo?", sino **"¿en quién funciona, en quién no, hasta cuándo, y a costa de qué?"**. Todo lo que has aprendido en el curso (métricas, calibración, validación) es justamente lo que te permite hacer esas preguntas con fundamento.
{% endhint %}

## 11.1 Sesgo y equidad: el modelo hereda el pasado

La intuición esencial: un modelo de ML no sabe medicina, sabe **datos históricos**. Aprende "lo que pasó", no "lo que debería pasar". Si el histórico contiene desigualdades —de acceso, de registro, de práctica clínica—, el modelo las aprende con la misma diligencia con que aprende todo lo demás, y las **repite a escala** con apariencia de objetividad.

Por dónde entra el sesgo, en la práctica:

* **Representación**: si un grupo está **infrarrepresentado** en los datos de entrenamiento (mujeres de edad avanzada, pacientes de un área rural, una etnia minoritaria), el modelo tiene menos ejemplos de los que aprender y rendirá **peor precisamente en ese grupo** —a menudo sin que nadie lo note, porque la métrica global sale bien.
* **Etiquetas**: la variable objetivo refleja la práctica pasada. Quien usó menos el sistema sanitario tiene menos diagnósticos registrados, y el modelo puede confundir "**sin diagnóstico**" con "**sano**".
* **Variables** _**proxy**_: columnas aparentemente neutras (código postal, tipo de centro, frecuentación) que codifican de tapadillo nivel socioeconómico o acceso a la sanidad.
* **Despliegue fuera de contexto**: un modelo entrenado en un hospital terciario aplicado tal cual en atención primaria, o al revés.

{% hint style="info" %}
**Concepto · Sesgo algorítmico**

Error **sistemático** de un modelo que perjudica (o favorece) a determinados grupos de pacientes. No suele nacer del algoritmo, sino de los **datos con que se entrenó**: quién está representado, cómo se midió y etiquetó, y qué desigualdades arrastraba la práctica que generó ese histórico. Un modelo con buena métrica global puede ser, a la vez, un mal modelo para un subgrupo concreto.
{% endhint %}

La consecuencia operativa es una sola, y es la que debes retener: **la métrica global esconde el sesgo; la evaluación por subgrupos lo destapa.** Un AUC global de 0,84 puede convivir con un rendimiento claramente peor en un subgrupo.

Por eso, ante cualquier modelo clínico, hay que pedir **sensibilidad, especificidad y calibración desglosadas** por sexo, tramo de edad, área o centro —los ejes que tengan sentido clínico en tu caso.

Nuestra cohorte **sintética** [`pacientes.csv`](https://drive.google.com/file/d/1Ku0j-sAf8Cr3FPT-DGm8v5p4h_2BmV5U/view?usp=drive_link) (20 000 pacientes generados por código, no reales) sirve para practicar la mecánica. Ya viste en U3 una pieza del asunto: la prevalencia de `evento_cv` varía por subgrupo (≈ 14 % en quienes nunca fumaron, ≈ 22 % en exfumadores, ≈ 28 % en fumadores activos), así que **el mismo umbral produce valores predictivos distintos en cada grupo**. El paso que falta es mirar también sensibilidad y especificidad **por subgrupo**, no solo el número global — y en unos datos reales nadie te garantiza que el desglose salga tan tranquilo como en unos sintéticos.

**🤖 Prompt para el asistente · Auditoría por subgrupos**

```
Sobre pacientes.csv (datos sintéticos del curso), entrena la regresión
logística de evento_cv como en U4 y evalúa en test. Después muéstrame una
tabla con sensibilidad, especificidad y VPP desglosadas por sexo y por tramo
de edad (<50, 50-70, >70), usando el mismo umbral para todos. Señala el
subgrupo con peor rendimiento y coméntame posibles causas.
```

{% hint style="warning" %}
**⚠️ Aviso: equidad no es ceguera**

Quitar la columna `sexo` del dataset **no elimina el sesgo**: otras variables correlacionadas pueden delatarla (efecto _proxy_), y a veces la variable sensible es **clínicamente necesaria** para predecir bien. La equidad no se consigue borrando columnas, sino **midiendo el rendimiento en cada grupo** y corrigiendo donde falla: más datos del grupo perjudicado, umbrales revisados o, si no hay arreglo, limitar el uso del modelo a las poblaciones donde funciona.
{% endhint %}

## 11.2 Validación clínica: "bueno en el test" no basta

En U3 aprendiste la evaluación honesta: partición de test intocable, métricas adecuadas al coste del error, calibración. Eso es **necesario pero no suficiente**, por una razón simple: el conjunto de test procede de **la misma población y la misma época** que el entrenamiento. Demuestra que el modelo generaliza a pacientes parecidos, no que funcione en tu hospital, con tus pacientes, el año que viene.

La validación de un modelo clínico tiene, por eso, **tres escalones**:

1. **Validación interna** (lo que hicimos en el curso): test aparte, validación cruzada. Descarta el sobreajuste, nada más.
2. **Validación externa**: probar el modelo en **otra población** —otro hospital, otra área, otra época— sin reentrenarlo. Es el equivalente a comprobar que un fármaco funciona fuera del ensayo que lo vio nacer. El rendimiento **casi siempre baja** al cambiar de población; la pregunta honesta no es "si baja", sino "cuánto, y si sigue siendo clínicamente útil".
3. **Validación en el tiempo**: comprobar que sigue funcionando **meses o años después**, porque el mundo cambia.

{% hint style="info" %}
**Concepto · Validación externa**

Evaluación de un modelo en una población **distinta e independiente** de la que lo entrenó (otro centro, otra área geográfica, otro periodo temporal). Es el estándar que separa "un modelo prometedor" de "una herramienta clínica creíble": responde a la pregunta que de verdad importa —**¿funciona con MIS pacientes?**— en lugar de asumirlo.
{% endhint %}

El tercer escalón merece nombre propio, porque es el gran olvidado: los modelos **envejecen**. Un modelo desplegado no es un proyecto cerrado, sino un **sistema vivo que se degrada** aunque nadie toque su código, porque los datos del mundo cambian. Esa degradación se llama **deriva** (_drift_), y la intuición cabe en dos frases:

* **Cambia lo que entra**: la población que atiendes ya no se parece a la del entrenamiento —cambia la demografía del área, se renueva el analizador del laboratorio y los valores vienen "movidos", una pandemia altera el perfil de quien acude a urgencias—. El modelo se encuentra operando en un terreno que no conoce.
* **Cambia la regla del juego**: la **relación** entre variables y desenlace ya no es la que el modelo aprendió —una nueva guía clínica hace tratar antes y el evento se vuelve menos frecuente para el mismo perfil de paciente; un fármaco nuevo cambia el pronóstico; se rediseña el circuito asistencial—. Las entradas parecen normales, pero la respuesta correcta ya es otra.

{% hint style="info" %}
**Concepto · Deriva (**_**drift**_**)**

Pérdida progresiva de rendimiento de un modelo en producción porque **el mundo se aleja de los datos con que se entrenó**: cambia la población que llega al modelo, o cambia la relación entre los datos y el desenlace (nuevas guías, nuevos tratamientos, nuevos circuitos). No es un fallo del código: es el **envejecimiento natural** de todo modelo, y la razón por la que necesita vigilancia y reevaluación periódicas.
{% endhint %}

Y un matiz que ya conoces de U3: la **calibración también envejece**. Un modelo puede seguir ordenando bien el riesgo y, sin embargo, dar probabilidades que ya no significan lo que dicen —un "20 %" de hace tres años puede no ser un 20 % hoy—. Por eso la recalibración periódica forma parte del mantenimiento normal de un modelo clínico.

La analogía sanitaria es exacta y te la puedes quedar: **farmacovigilancia**. La aprobación de un fármaco no es el final del escrutinio, sino el comienzo de una vigilancia continua; con un modelo, igual: alguien tiene que **comparar periódicamente sus predicciones con los desenlaces reales** y decidir cuándo recalibrar, reentrenar o retirar.

Cómo se industrializa esa vigilancia es una disciplina propia (se llama MLOps) que queda fuera de este curso; lo que un profesional sanitario debe exigir es más sencillo: que esa vigilancia **exista**, tenga métricas definidas y tenga **un responsable con nombre**.

{% hint style="success" %}
**💡 Idea clave**

**"Bueno en el test" es el principio, no el final.** Un modelo clínico creíble acumula tres evidencias: validación **interna** (no se engaña a sí mismo), validación **externa** (funciona fuera de casa) y **vigilancia en el tiempo** (alguien comprueba que sigue funcionando, porque el modelo envejece). Un modelo sin plan de vigilancia es como un fármaco sin farmacovigilancia: puede ir bien, pero nadie se enteraría de lo contrario.
{% endhint %}

## 11.3 Privacidad: por qué este curso es 100 % sintético

Habrás notado que cada unidad repite la misma coletilla: **todos los datos del curso son sintéticos**, generados por código. No es una limitación: es una decisión, y es en sí misma contenido formativo.

Los datos de salud son de las categorías **más sensibles** que existen, y la regla profesional es tajante: no se trabaja con datos de pacientes sin las debidas garantías éticas y legales — tampoco para "solo probar un modelo".

Los datos sintéticos, como técnica, tienen su propio balance:

**✅ Fortalezas**

* **Riesgo de privacidad nulo**: no hay ningún paciente detrás; nada que reidentificar.
* **Compartibles y reproducibles**: cualquiera puede ejecutar los notebooks sin permisos ni comités, y los resultados son replicables.
* **Perfectos para aprender la mecánica**: métricas, validación, sesgos y modelos se comportan como con datos reales.

**⚠️ Debilidades y límites**

* **No sustituyen a los datos reales para validar**: un modelo destinado a uso asistencial debe evaluarse con datos reales de la población diana, con sus permisos correspondientes.
* Pueden **no contener las rarezas** del mundo real (errores de registro extraños, casos atípicos, interacciones no previstas por el generador).
* **Ninguna conclusión clínica** puede extraerse de ellos: son un gimnasio, no un estudio.

**Campo de aplicación clínica.** Formación (este curso), prototipado de ideas, desarrollo y prueba de código antes de solicitar acceso a datos reales, y documentación/demos que deban circular sin restricciones.

Cuando los datos son reales, las dos herramientas básicas de protección conviene conocerlas por su nombre — y conocer su **fragilidad**:

{% hint style="info" %}
**Concepto · Anonimización y seudonimización**

**Seudonimizar** es sustituir los identificadores directos (nombre, número de historia) por un código, conservando en algún lugar la tabla que permite revertirlo: el dato **sigue siendo personal** y sigue exigiendo protección. **Anonimizar** es romper el vínculo con la persona de forma **irreversible**… en teoría. En la práctica, la anonimización es un **análisis de riesgo**, no un conjuro: hay que evaluar cuán fácil sería reidentificar, no dar por hecho que quitando el nombre basta.
{% endhint %}

{% hint style="warning" %}
**⚠️ Aviso: la reidentificación es más fácil de lo que parece**

Quitar el nombre no anonimiza. Combinaciones de **cuasi-identificadores** —edad, sexo, código postal, fechas de ingreso, un diagnóstico poco frecuente— pueden hacer **única** a una persona dentro de una tabla, y cruzarla con otras fuentes permite reidentificarla. Cuanto más rica y detallada es la tabla (y las tablas clínicas lo son), mayor el riesgo. De ahí el principio de **minimización**: usa solo las variables y la granularidad **estrictamente necesarias** para la pregunta que quieres responder.
{% endhint %}

Y el punto donde esto conecta con tu práctica inmediata, porque ya lo viviste en U9: los **modelos por API** (los asistentes conversacionales, OpenRouter y compañía). Todo lo que escribes en un chat público o envías por una API **sale de tu entorno y viaja a un tercero**.

Pegar una nota clínica real en un chatbot para "que me la resuma" es, técnicamente, **enviar datos de un paciente a una empresa externa**, con independencia de la buena intención.

{% hint style="danger" %}
**⚠️ La regla del curso es la regla profesional**

A una API pública, **solo datos sintéticos, agregados o debidamente anonimizados**; el uso de datos identificables exige un marco legal y técnico específico (contratos, entornos controlados, despliegues locales como los de Hugging Face que vimos en U9) — y esa valoración no la hace un clínico en solitario, sino con el delegado de protección de datos y los circuitos de su centro.
{% endhint %}

## 11.4 El marco regulatorio, en titular

No estás solo ante estas exigencias: existe un **marco regulatorio europeo** que apunta exactamente en la dirección de todo lo anterior. A nivel de titular —que es el nivel de esta unidad—, dos piezas conviven:

* Un **reglamento europeo de inteligencia artificial** (el conocido como _AI Act_), que clasifica los sistemas de IA **por nivel de riesgo** y, para los usos de alto riesgo —entre los que se cuentan aplicaciones sanitarias—, exige cosas que ya te sonarán: **gestión de riesgos**, **calidad y gobernanza de los datos**, **transparencia y documentación**, y **supervisión humana** efectiva.
* El **marco de producto sanitario**: el software que apoya decisiones diagnósticas o terapéuticas puede tener la consideración de **producto sanitario**, con la evaluación y certificación que eso conlleva antes de su uso asistencial, y obligaciones de seguimiento después.

Lo revelador es que la regulación llega, por la vía jurídica, a las **mismas conclusiones** a las que este curso ha llegado por la vía técnica: clasifica por riesgo, valida, documenta, vigila en el tiempo y deja decidir al humano. Si has interiorizado las unidades anteriores, el espíritu de la norma no te resultará extraño.

Aquí nos detenemos a propósito. **El detalle legal —qué sistemas caen en qué categoría, qué obligaciones concretas aplican y cómo se articula todo en España— corresponde a la sesión legal del Módulo III de este curso**, impartida por especialistas en derecho sanitario y de la IA. Esta sección solo pretende que llegues a ella con el mapa técnico puesto: sabiendo ya **por qué** la norma pide lo que pide.

## 11.5 Responsabilidad y supervisión humana: la IA asiste, tú decides

El principio que ordena todo lo demás: **la IA asiste; el profesional decide y responde**. Un modelo no firma historias, no informa a la familia y no comparece ante un comité: propone, prioriza, alerta o resume, y la decisión clínica —con su responsabilidad— sigue siendo humana. Esto no es una cortesía retórica: es la base sobre la que se asienta el marco regulatorio que acabamos de ver, y la razón por la que la "supervisión humana" aparece en él como exigencia y no como sugerencia.

Para que esa supervisión sea real y no un trámite, tres condiciones:

* **Capacidad de discrepar.** Existe una tendencia humana bien conocida a aceptar la sugerencia de la máquina, sobre todo con prisa y carga asistencial (el llamado **sesgo de automatización**). La supervisión solo existe si el profesional conserva el hábito —y el respaldo institucional— de decir "no estoy de acuerdo con el modelo" y actuar en consecuencia.
* **Derecho a explicación.** Ante una sugerencia relevante, profesional y paciente deben poder saber **por qué**: qué variables han pesado en este caso concreto. Aquí conecta lo que viste en U5 (SHAP) y una lección práctica del propio curso: en nuestra cohorte sintética, la regresión logística (AUC ≈ 0,84) superó al Random Forest (≈ 0,83) — cuando el modelo interpretable empata o gana, la explicabilidad **sale gratis**; renunciar a ella exige justificación, no al revés.
* **Información y consentimiento.** El paciente tiene derecho a saber que en su proceso interviene un sistema automatizado, en los términos y circuitos que marque el centro y la normativa (de nuevo: el detalle, en el Módulo III). Y el profesional, a un **circuito de notificación de fallos** del sistema, análogo al de incidentes de seguridad del paciente: un modelo que se equivoca de forma sistemática es un evento notificable, no una anécdota.

{% hint style="danger" %}
**⚠️ Aviso: la facilidad no diluye la responsabilidad**

Cerramos el círculo que abrimos en U9: cuanto más fácil es usar un modelo —tres líneas de código, un chat, una API—, **mayor** es la tentación de saltarse la validación, la privacidad y la supervisión. La facilidad de uso no reduce tu responsabilidad profesional: **la aumenta**, porque elimina las barreras que antes obligaban a pararse a pensar. El criterio que has construido en este curso es, exactamente, ese punto de parada.
{% endhint %}

## 11.6 El checklist: qué preguntar antes de confiar en un modelo clínico

Todo lo anterior, condensado en las preguntas que deberías poder hacer —y alguien debería poder responder— antes de incorporar cualquier sistema de IA a tu práctica:

1. **¿Con qué datos se entrenó** y en qué se parecen (edad, sexo, comorbilidad, ámbito) a **mis** pacientes?
2. **¿Se ha validado externamente**, en una población distinta de la que lo entrenó — idealmente en un entorno como el mío?
3. **¿Qué métricas reporta y desglosadas por subgrupos** (sexo, edad, área), o solo una cifra global?
4. **¿Está calibrado en mi población?** Cuando dice "20 % de riesgo", ¿es de verdad un 20 %?
5. **¿Qué errores comete y cuánto cuestan?** ¿Se eligió el umbral pensando en el coste clínico del falso negativo frente al falso positivo, y quién lo decidió?
6. **¿Puedo saber por qué** sugiere lo que sugiere en un paciente concreto (explicación caso a caso)?
7. **¿Quién lo vigila en el tiempo?** ¿Con qué periodicidad se reevalúa y recalibra, qué umbral dispara una revisión y quién es el responsable?
8. **¿Dónde van los datos que introduzco?** ¿Se procesan en local o viajan a terceros, con qué base legal y con qué minimización?
9. **¿Cuál es su estatus regulatorio?** ¿Está certificado para el uso que le estoy dando, o lo estoy usando fuera de su indicación?
10. **¿Queda claro que la decisión final es mía**, y existe un circuito para discrepar del sistema y notificar sus fallos?

{% hint style="info" %}
**Concepto · Una regla de lectura**

Si la respuesta a varias de estas diez preguntas es "no se sabe", **esa es la respuesta**: no hace falta encontrar un fallo concreto para desconfiar, basta con que nadie pueda contestarte.
{% endhint %}

## 11.7 Qué llevarte

* **El modelo hereda el sesgo del histórico** y la métrica global lo esconde: exige siempre la evaluación **por subgrupos** (sensibilidad, especificidad y calibración por sexo, edad, área).
* **"Bueno en el test" no basta**: pide validación **externa**, calibración comprobada y un plan de **vigilancia en el tiempo**, porque los modelos **envejecen** (deriva) — como los fármacos tienen farmacovigilancia.
* **Privacidad por diseño**: sintéticos para aprender y prototipar; la anonimización es frágil (reidentificación); minimiza los datos; y a una API de terceros, **jamás** datos identificables sin marco legal y técnico (U9).
* Existe un **marco europeo** que clasifica por riesgo y exige supervisión humana, transparencia y gestión de riesgos; el detalle legal te espera en la **sesión legal del Módulo III**.
* **La IA asiste; tú decides y respondes.** El checklist de 11.6 es tu instrumento de trabajo: llévatelo puesto.

Y con esta lente termina la parte de Machine Learning del curso. Ya sabes qué familia de modelos encaja en cada problema (U2–U8), dónde están los modelos ya hechos y cómo usarlos (U9), cómo dirigir a la IA como copiloto (U10) y —desde hoy— **qué exigirle a todo ello antes de fiarte**. El siguiente paso del programa toma el relevo justo donde lo hemos dejado: la **sesión legal del Módulo III**, donde estos principios se convierten en obligaciones concretas. Llegas a ella con lo más difícil ya hecho: el criterio.
