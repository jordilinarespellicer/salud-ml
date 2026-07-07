---
description: >-
  Puesta a punto opcional para quien no ha programado nunca: qué es Google
  Colab, el mínimo de Python para leer código sin miedo y por qué en este curso
  el código lo escribe un asistente de IA mientras tú pones el criterio clínico.
---

# U0 · Puesta a punto — Google Colab y Python

{% hint style="info" %}
**Concepto · Unidad de puesta a punto (opcional)**

Esta unidad es una **red de seguridad**, no un examen de acceso. Está pensada para quien no ha programado nunca y quiere llegar a la primera práctica con el terreno pisado.

Si ya has usado Google Colab y sabes leer cuatro líneas de Python, puedes saltártela sin remordimientos y volver solo cuando algo te chirríe: estará siempre aquí.
{% endhint %}

Empecemos por lo más importante: **en este curso no vas a aprender a programar, y no pasa absolutamente nada**. Vas a aprender a *leer* código —que es muchísimo más fácil que escribirlo— y a *pedirlo* bien.

Es la misma diferencia que hay entre interpretar un electrocardiograma y diseñar el electrocardiógrafo: para usar el aparato con criterio te basta lo primero.

Todo lo que viene a continuación son **ideas, no detalle**. Nada de memorizar sintaxis ni de estudiar la noche antes: si dentro de tres unidades no recuerdas cómo se escribía un bucle, no habrás perdido nada, porque el código te lo escribirá un asistente de IA.

Lo que sí te llevarás de aquí es algo más valioso: la tranquilidad de mirar una pantalla llena de código y pensar «ah, ya veo por dónde va esto».

## 0.1 Qué es Google Colab y por qué lo usamos

[**Google Colab**](https://colab.research.google.com/) es un servicio **gratuito** de Google que ejecuta **cuadernos de Python** (*notebooks*) directamente **en el navegador**.

Eso significa cuatro cosas que lo hacen perfecto para un curso como este:

- **No hay que instalar nada.** Ni Python, ni librerías, ni programas extraños. Si tu ordenador abre YouTube, puede seguir este curso. El código se ejecuta en un ordenador de Google, en la nube; tu navegador solo muestra el resultado.
- **Es gratis.** La versión gratuita sobra para todo lo que haremos.
- **Viene con el instrumental ya puesto.** Las librerías de datos que usaremos (pandas, numpy, matplotlib…) están preinstaladas, como un quirófano montado antes de que llegues.
- **Incluye GPU gratis.** Un tipo de procesador muy potente que pediremos prestado cuando lleguemos al *deep learning*. De momento, basta con saber que existe y que no cuesta nada.

Si te ayuda una analogía: Colab es a la programación lo que el laboratorio central es a la analítica. Tú pides la prueba e interpretas el resultado; no necesitas montar el espectrofotómetro en tu consulta.

### Abrir un cuaderno y moverse por él

Los cuadernos del curso se abren con **un clic en un enlace**: se despliegan en una pestaña del navegador con tu cuenta de Google.

Si quieres conservar tus cambios, usa **Archivo → Guardar una copia en Drive**: desde ese momento tienes tu propia copia, que **se guarda sola en tu Google Drive** como un documento más. Imposible perderla, imposible estropear el original.

Un *notebook* es un documento hecho de **celdas** que se leen y se ejecutan **de arriba abajo**, igual que una historia clínica se lee en orden cronológico. Hay dos tipos:

- **Celdas de texto** — explicaciones normales y corrientes, con títulos y negritas. En los cuadernos del curso son mayoría: te van contando qué está pasando y por qué, como los comentarios al margen de un protocolo.
- **Celdas de código** — contienen Python. Se ejecutan pulsando el botón **▶** que aparece a su izquierda, o con **Mayús + Enter**. El resultado (un número, una tabla, una gráfica) aparece justo debajo de la celda.

Y el manejo mínimo, que cabe en tres líneas:

- **Ejecutar una celda:** **▶** o **Mayús + Enter** (ejecuta y salta a la siguiente).
- **Ejecutar el cuaderno entero:** menú **Entorno de ejecución → Ejecutar todo**. Es la opción estrella de este curso: reproduce todo el análisis de arriba abajo, en orden, sin que tengas que tocar nada más.
- **El orden importa:** las celdas comparten memoria, así que una variable creada arriba existe más abajo. Si ejecutas celdas salteadas puede aparecer algún error extraño; no es grave, es solo desorden.

Un último detalle tranquilizador: si dejas Colab un rato sin usar, la sesión **se desconecta** y las variables se olvidan (el texto y el código del cuaderno, no: esos quedan guardados en Drive). La solución es siempre la misma: reconectar y **Ejecutar todo**.

{% hint style="success" %}
**💡 Idea clave**

En Colab **no puedes romper nada**. Ni tu ordenador (el código corre en la nube), ni el material del curso (tú trabajas sobre tu copia de Drive).

Ante cualquier error raro o resultado incoherente, el botón de reinicio universal es **Entorno de ejecución → Ejecutar todo**: el cuaderno vuelve a su ser en unos segundos. Trastea sin miedo.
{% endhint %}

## 0.2 El mínimo de Python para leer código sin miedo

**Python** es el lenguaje en el que están escritos todos los cuadernos, y lo elegimos precisamente porque se lee casi como inglés sencillo.

Nuestra meta aquí no es escribirlo, sino **leerlo**: exactamente como con el inglés científico, que la mayoría leemos con soltura aunque no lo hablemos fino. Cuatro piezas bastan para descifrar el 90 % de lo que verás.

### Variables: ponerle nombre a un dato

Una **variable** es una etiqueta con un valor guardado detrás. Se crea con el signo `=`:

```python
edad = 54          # un número entero
glucemia = 118.5   # un número con decimales
sexo = "F"         # un texto (va entre comillas)
```

Eso es todo: `edad` vale 54 y podemos usarla más abajo. Lo que va después de `#` es un **comentario**, una nota al margen que Python ignora y que está ahí solo para el lector humano.

Fíjate en que hay **tipos** de datos —enteros, decimales, texto— pero Python los distingue solo, sin que tengas que declarar nada.

### Listas: varios valores en fila

Una **lista** agrupa valores entre corchetes, como una columna de resultados de laboratorio:

```python
glucemias = [92, 110, 145, 128, 99]   # mg/dL, datos inventados de ejemplo
```

### Un bucle: repetir algo para cada elemento

Un **bucle** `for` recorre la lista y hace lo mismo con cada valor. Léelo en voz alta y verás que se explica solo:

```python
for g in glucemias:
    if g >= 126:
        print(g, "→ por encima del umbral diagnóstico")
```

En cristiano: «para cada glucemia `g` de la lista, si es mayor o igual a 126 mg/dL (el umbral clásico de diabetes en ayunas), imprímela con un aviso». Aquí saltarían 145 y 128.

Fíjate en el detalle visual: lo que va **sangrado** (desplazado a la derecha) es lo que ocurre «dentro» del bucle. Esa sangría es la forma que tiene Python de anidar ideas, como los subapartados de un informe.

### Una función: un protocolo con nombre

Una **función** empaqueta un cálculo para reutilizarlo. Es literalmente un protocolo: se define una vez y se aplica mil veces. El ejemplo canónico en nuestro mundo, el IMC:

```python
def imc(peso_kg, talla_m):
    return peso_kg / talla_m ** 2

imc(70, 1.65)   # → 25.7
```

`def` define la función, los paréntesis llevan los datos de entrada y `return` devuelve el resultado (`**` es «elevado a»). A partir de ahí, `imc(70, 1.65)` responde 25.7 sin volver a escribir la fórmula.

Las librerías que veremos ahora son exactamente esto: **miles de funciones ya escritas y probadas por otros**, listas para llamar.

{% hint style="success" %}
**💡 Idea clave**

**No hay que memorizar nada de esto.** El objetivo es que `for`, `if`, `def` o los corchetes de una lista te suenen como te suenan «TA», «FC» o «Rx»: abreviaturas de ideas simples que se descifran al vuelo. Se lee y se entiende; escribirlo ya es cosa del asistente.
{% endhint %}

## 0.3 Las librerías que verás una y otra vez (en modo idea)

Una **librería** es un conjunto de funciones ya hechas que se «importan» al principio del cuaderno (de ahí las líneas `import` que abren casi todos).

No hace falta saber qué contienen por dentro; basta con saber **para qué sirve cada una**, igual que no hace falta conocer la óptica del microscopio para saber cuándo pedirlo. Las tres que aparecerán una y otra vez:

- **pandas** — trabaja con **tablas de datos** (los famosos `DataFrame`). Piensa en una hoja de cálculo con superpoderes: carga un CSV de 20 000 pacientes, filtra, agrupa, resume y cruza columnas en una línea. Es la librería que más verás, con diferencia.
- **numpy** — el motor de **cálculo numérico** que trabaja por debajo de casi todo: medias, redondeos, operaciones con miles de números a la vez. Suele aparecer como `np` en el código.
- **matplotlib y seaborn** — las encargadas de las **gráficas**: histogramas de edades, nubes de puntos de IMC frente a tensión, curvas de evolución. Seaborn es, en esencia, un matplotlib con mejor gusto estético.

Más adelante se sumará una cuarta protagonista, **scikit-learn**, que es la caja de herramientas de Machine Learning propiamente dicha; la presentaremos con calma cuando toque entrenar el primer modelo.

### Mini-ejemplo: asomarse a una tabla de pacientes

Así se carga una tabla y se le echa el primer vistazo. Dos líneas, y las dos se entienden solas:

```python
import pandas as pd                 # importamos pandas (con su apodo habitual)

df = pd.read_csv("pacientes.csv")   # cargamos la tabla en una variable llamada df
df.head()                           # muestra las 5 primeras filas
```

Al ejecutarlo aparece debajo una tabla con las primeras filas de `pacientes.csv`, nuestro dataset **sintético** de cabecera (20 000 pacientes generados artificialmente, sin ninguna persona real detrás): columnas como `edad`, `sexo`, `imc`, `ta_sistolica`, `glucemia`, `colesterol_total` o `tabaquismo`.

Ese gesto —cargar y mirar con `head()`— será el primer acto reflejo ante cualquier dato durante todo el curso.

{% hint style="info" %}
**Concepto · DataFrame**

La tabla de pandas. Cada **fila** es un caso (aquí, un paciente) y cada **columna** una variable (edad, IMC, glucemia…). Exactamente la misma estructura que la sábana de datos de un estudio o una hoja de Excel bien organizada.

Cuando en el código veas `df`, casi siempre es un DataFrame; el nombre es pura costumbre.
{% endhint %}

## 0.4 El principio liberador: la IA escribe el código, tú pones el criterio

Y aquí llega la idea que hace posible este curso para un profesional sanitario sin formación técnica: **en 2026 el valor ya no está en escribir código de memoria, sino en tener criterio clínico y saber pedirle el código a un asistente de IA**.

Los asistentes actuales (Claude, ChatGPT, Gemini y compañía) escriben con soltura el código de análisis de datos que hace diez años exigía un programador. Lo que **no** pueden poner es lo que tú traes de serie: saber si una pregunta tiene sentido clínico, si una variable está mal medida o si un resultado es demasiado bonito para ser verdad.

El reparto de papeles queda así, y es un buen trato:

- **El asistente**: escribe el código, lo corrige cuando falla y te explica línea a línea lo que hace.
- **Tú**: decides qué preguntar, compruebas que la respuesta tiene sentido clínico y detectas lo que chirría (¿una glucemia de 800 en un paciente «sano»? ¿un IMC de 3?).

En la práctica, todo empieza por saber **pedir**. Un buen prompt es como una buena interconsulta: contexto, datos disponibles y qué necesitas. Por ejemplo:

> Tengo un fichero `pacientes.csv` con datos **sintéticos** de 20 000 pacientes: edad, sexo, IMC, tensión arterial, glucemia, colesterol, tabaquismo y si hubo evento cardiovascular. Escríbeme el código Python con pandas para cargarlo, ver las primeras filas y obtener un resumen estadístico de cada columna. Explícame cada línea en una frase, como a alguien que no programa.

El asistente devolverá un bloque de código muy parecido al del apartado anterior (con un `df.describe()` de regalo para el resumen estadístico), listo para pegar en una celda de Colab y ejecutar. Si algo falla, se le pega el mensaje de error tal cual y lo arregla.

Ese ciclo —**pedir, ejecutar, leer, repreguntar**— es la forma de trabajar de este curso, y lo practicaremos desde el primer día.

{% hint style="warning" %}
**⚠️ Aviso**

Nunca pegues **datos reales de pacientes** en un asistente de IA ni los subas a servicios en la nube sin las garantías de tu organización: es información especialmente protegida.

En este curso no habrá ocasión de hacerlo, porque **todos los datos son sintéticos**, generados artificialmente para parecerse a los reales sin corresponder a ninguna persona. Pero la costumbre conviene traerla puesta desde ya.
{% endhint %}

## 0.5 Cómo usar los notebooks del curso

Cada unidad tiene su cuaderno de Colab, y todos siguen la misma coreografía pensada para que la parte técnica sea invisible:

- **La primera celda genera los datos sintéticos.** No hay que descargar ficheros, ni subir nada, ni pedir permisos: al ejecutarla, el propio cuaderno fabrica su `pacientes.csv` (o el dataset que toque) en la sesión. Siempre funciona, en cualquier ordenador.
- **Ábrelo y dale a Entorno de ejecución → Ejecutar todo.** En unos segundos tendrás el cuaderno completo ejecutado, con todas sus tablas y gráficas. A partir de ahí, léelo con calma de arriba abajo, que es donde está el aprendizaje.
- **Trastea.** Cambia un número, vuelve a ejecutar la celda y mira qué pasa. Es la mejor manera de aprender y, como ya sabes, no puedes romper nada: **Ejecutar todo** lo restaura siempre.
- **Guarda tu copia en Drive** si quieres conservar tus experimentos y anotaciones de una sesión a otra.

{% hint style="info" %}
**Concepto · La coreografía de cada notebook, en una frase**

Datos sintéticos autogenerados en la primera celda → **Ejecutar todo** → leer de arriba abajo → trastear sin miedo → guardar tu copia si quieres conservarla. Cuatro gestos que se repiten en todas las unidades del curso.
{% endhint %}

{% hint style="success" %}
**🔬 Práctica en Colab — `U00_Bienvenida_Colab.ipynb`**

Tu primer cuaderno: un paseo guiado por Colab y por todo lo que has leído aquí —celdas, variables, la función del IMC, pandas y tu primera tabla de pacientes—.

**Su primera celda genera los datos sintéticos** del curso: no hay que descargar ni configurar nada. Ábrelo, pulsa *Ejecutar todo* y dedícale veinte minutos tranquilos.

[Abrir en Colab](PENDIENTE_DRIVE)
{% endhint %}

## Qué llevarte

- **Colab es Python en el navegador**: gratis, sin instalar nada, con GPU disponible y con tus cuadernos guardados solos en Google Drive.
- Un cuaderno se lee **de arriba abajo**; ante cualquier duda o error, **Entorno de ejecución → Ejecutar todo**. En Colab no se puede romper nada.
- Python **se lee mejor de lo que se escribe**: variables, listas, `for` y `def` son abreviaturas de ideas simples, como las siglas clínicas. Nada que memorizar.
- En modo idea: **pandas** = tablas, **numpy** = números, **matplotlib/seaborn** = gráficas. Y `df.head()` como primer acto reflejo ante cualquier dato.
- **El asistente de IA escribe el código; tú aportas el criterio clínico** y aprendes a pedir y a leer. Eso sí: jamás con datos reales de pacientes, siempre con nuestros datos sintéticos.

Con la puesta a punto hecha, en la siguiente unidad empieza el curso de verdad: qué es exactamente eso del *machine learning* y por qué a un modelo no se le programa, sino que **se le entrena con datos** — datos como los de nuestro `pacientes.csv`.
