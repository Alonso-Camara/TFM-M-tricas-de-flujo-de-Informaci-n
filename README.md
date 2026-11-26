# Métricas de flujo de información en modelos neuronales computacionales

Código asociado al Trabajo Fin de Máster *«Análisis y comparación de métricas de flujo de información en modelos neuronales computacionales: modelos de conductancia y modelos fenomenológicos»*.

El repositorio contiene las simulaciones y los análisis que respaldan los resultados de la memoria. Se estiman la Información Mutua (IM), la Granger Causality (GC) y la Transfer Entropy (TE) sobre sistemas autorregresivos, modelos neuronales (Hodgkin-Huxley e Izhikevich) y el modelo del sistema electromotor del pez de pulso eléctrico. Este documento explica cómo preparar un entorno local para ejecutar los notebooks.

---

## Estructura del repositorio

```
.
├── scripts/
│   └── Modelos_neuronales.py      # Funciones de simulación de Hodgkin-Huxley e Izhikevich
└── notebooks/
    ├── *.ipynb                    # Experimentos y resultados de la memoria
    ├── infodynamics.jar           # Librería JIDT 1.6.1 (Java)
    └── IDTxl-master/
        └── IDTxl-master/          # Código fuente de IDTxl (descargado del repositorio oficial)
```

- **`scripts/`** contiene únicamente un par de funciones auxiliares para simular los modelos de Hodgkin-Huxley e Izhikevich. Los notebooks las importan como módulo (`import Modelos_neuronales as mn`).
- **`notebooks/`** contiene todos los experimentos y resultados presentados en la memoria. Además incluye los dos componentes externos necesarios para estimar la TE y ejecutar el análisis multivariado: el archivo `infodynamics.jar` (JIDT) y la carpeta `IDTxl-master`.

> Los notebooks deben ejecutarse **desde la propia carpeta `notebooks/`**: las rutas a `infodynamics.jar`, a `IDTxl-master` y a `scripts/` se construyen de forma relativa a ella.

---

## Entorno de referencia

Todos los resultados se han obtenido con la siguiente configuración. Versiones distintas probablemente funcionen, pero son las únicas verificadas.

| Componente | Versión |
|---|---|
| Sistema operativo | Windows 11 (64 bits) |
| Distribución de Python | Anaconda |
| Python | 3.13.5 |
| Java (JDK) | Azul Zulu 11 · OpenJDK 11.0.29 LTS (64 bits) |
| JPype (`jpype1`) | 1.7.1 |
| JIDT (`infodynamics.jar`) | 1.6.1 |
| IDTxl | Código fuente de la rama `master` del repositorio oficial |
| NumPy | 2.1.3 |
| SciPy | 1.15.3 |
| scikit-learn | 1.6.1 |
| NetworkX | 3.4.2 |

---

## Por qué hace falta Java

Las dos herramientas centrales del trabajo dependen de Java:

- **JIDT** (*Java Information Dynamics Toolkit*, Lizier, 2014) implementa los estimadores de Kraskov-Stögbauer-Grassberger (KSG) usados para la IM y la TE. Está escrita en Java y se distribuye como un único archivo `.jar`.
- **IDTxl** (*Information Dynamics Toolkit xl*, Wollstadt et al., 2019) es una librería de Python para el análisis multivariado. Internamente usa JIDT como motor de cálculo.

Para llamar a código Java desde Python se usa **JPype**, que arranca una máquina virtual de Java (JVM) dentro del propio proceso de Python. Por tanto, el equipo necesita:

1. un JDK de Java instalado;
2. el paquete `jpype1` en Python;
3. el archivo `infodynamics.jar`, que ya se incluye en `notebooks/`.

---

## Instalación paso a paso

### 1. Instalar Java (JDK)

Se necesita un **JDK de 64 bits** (la arquitectura debe coincidir con la de Python). El trabajo se ha ejecutado con **Azul Zulu 11**, pero cualquier JDK 11 o posterior debería funcionar (por ejemplo, Eclipse Temurin u Oracle JDK).

- Azul Zulu: <https://www.azul.com/downloads/?package=jdk>
- Eclipse Temurin: <https://adoptium.net/>

Tras la instalación, conviene definir la variable de entorno `JAVA_HOME` apuntando a la carpeta del JDK (no a `bin`). Por ejemplo, en Windows:

```
JAVA_HOME = C:\Program Files\Zulu\zulu-11
```

JPype usa `JAVA_HOME` (o, si no existe, el registro de Windows) para localizar la JVM. Para comprobar qué JVM encontrará:

```python
import jpype
print(jpype.getDefaultJVMPath())
# p. ej. C:\Program Files\Zulu\zulu-11\bin\server\jvm.dll
```

### 2. Crear el entorno de Python

Se recomienda un entorno dedicado. Desde *Anaconda Prompt*:

```bash
conda create -n tfm-info python=3.13
conda activate tfm-info
```

### 3. Instalar los paquetes de Python

**Paquetes científicos básicos.** Se usan de forma habitual en todos los notebooks:

```bash
pip install numpy scipy pandas matplotlib tqdm scikit-learn jupyter
```

`scikit-learn` proporciona `mutual_info_regression` (estimador de IM de vecinos más cercanos usado como referencia) y `LinearRegression`. Basta con instalar el paquete; no requiere ninguna configuración adicional.

**Puente Python–Java:**

```bash
pip install jpype1
```

**Dependencias de IDTxl.** Solo son necesarias en los notebooks que usan IDTxl:

```bash
pip install networkx
```

`networkx` lo necesita `idtxl.visualise_graph` (función `plot_network`). IDTxl tiene otras dependencias opcionales (`pyopencl`, `numba`, `h5py`, `ete3`) que **no** se usan en este trabajo.

### 4. JIDT

No requiere instalación. El archivo `notebooks/infodynamics.jar` corresponde a la **versión 1.6.1** de JIDT, publicada en <https://github.com/jlizier/jidt>. Si se desea sustituirlo, basta con descargar la distribución de JIDT y copiar su `infodynamics.jar` en `notebooks/`.

### 5. IDTxl

IDTxl no se instala como paquete: se usa directamente desde su código fuente, incluido en `notebooks/IDTxl-master/IDTxl-master` y descargado de <https://github.com/pwollstadt/IDTxl>. Los notebooks añaden esa carpeta al `sys.path` para poder importarlo (véase la sección siguiente).

---

## Configuración inicial de los notebooks

Cada notebook comienza con unas celdas de configuración. Sus dos variantes son las siguientes.

### Notebooks que usan JIDT

```python
import sys
import os
from pathlib import Path

# Funciones de simulación de la carpeta scripts/
ruta_scripts = Path.cwd().parent / "scripts"
sys.path.append(str(ruta_scripts))
import Modelos_neuronales as mn

# Arranque de la JVM con JIDT
import jpype
from jpype import JPackage

jar_path = os.path.abspath("infodynamics.jar")
jpype.startJVM(classpath=[jar_path])

kraskov = JPackage("infodynamics.measures.continuous.kraskov")
teCalc  = kraskov.TransferEntropyCalculatorKraskov()              # TE
cteCalc = kraskov.ConditionalTransferEntropyCalculatorKraskov()   # TE condicional
miCalc  = kraskov.MutualInfoCalculatorMultiVariateKraskov1()      # IM (algoritmo 1 de KSG)
```

### Notebooks que usan IDTxl

Se arranca primero la JVM con JIDT, igual que en el caso anterior, y después se añade IDTxl al `sys.path`:

```python
import sys
import math
from pathlib import Path
import numpy as np

# Ruta a IDTxl relativa a la carpeta notebooks/
ruta_idtxl = Path.cwd() / "IDTxl-master" / "IDTxl-master"
sys.path.insert(0, str(ruta_idtxl))

# Compatibilidad: NumPy >= 2.0 eliminó np.math, que IDTxl todavía utiliza
np.math = math

from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data
from idtxl.visualise_graph import plot_network
```

> **Atención:** en los notebooks originales esta ruta aparece escrita como ruta absoluta del equipo del autor (`sys.path.insert(0, r"C:\Users\...\IDTxl-master\IDTxl-master")`). Para ejecutarlos en otro equipo, sustitúyela por la versión relativa anterior o por la ruta donde se encuentre la carpeta en tu máquina.

---

## Problemas frecuentes

| Síntoma | Causa probable y solución |
|---|---|
| `JVMNotFoundException` o `No JVM shared library file (jvm.dll) found` | JPype no encuentra Java. Instala un JDK de 64 bits y define `JAVA_HOME`. Reinicia Jupyter después de cambiar variables de entorno. |
| `OSError: JVM is already started` | La JVM solo puede arrancarse **una vez por kernel**. Ejecuta la celda de configuración una única vez o reinicia el kernel (*Kernel → Restart*). |
| `OSError: JVM cannot be restarted` | Igual que el anterior: tras cerrar la JVM es obligatorio reiniciar el kernel. |
| `java.lang.ClassNotFoundException` / `TypeError: Package infodynamics... is not callable` | La JVM se arrancó sin `infodynamics.jar` en el *classpath*. Comprueba que el notebook se ejecuta desde `notebooks/` y que `os.path.abspath("infodynamics.jar")` apunta a un archivo existente. |
| `AttributeError: module 'numpy' has no attribute 'math'` | Falta la línea `np.math = math` antes de importar IDTxl (incompatibilidad de IDTxl con NumPy 2.x). |
| `ModuleNotFoundError: No module named 'idtxl'` | La ruta añadida a `sys.path` no es la carpeta que contiene el subdirectorio `idtxl/`. Revisa la doble carpeta `IDTxl-master/IDTxl-master`. |
| `ModuleNotFoundError: No module named 'Modelos_neuronales'` | El notebook no se está ejecutando desde `notebooks/`, por lo que `Path.cwd().parent / "scripts"` no apunta a la carpeta correcta. |

---

## Licencias y referencias

JIDT e IDTxl se redistribuyen en este repositorio según los términos de sus licencias originales (GNU GPL v3), que se conservan en sus respectivas carpetas. Si utilizas estas herramientas, cita los trabajos originales:

- J. T. Lizier, «JIDT: An information-theoretic toolkit for studying the dynamics of complex systems», *Frontiers in Robotics and AI*, 1:11, 2014.
- P. Wollstadt, J. T. Lizier, R. Vicente, C. Finn, M. Martínez-Zarzuela, P. Mediano, L. Novelli y M. Wibral, «IDTxl: The Information Dynamics Toolkit xl: a Python package for the efficient analysis of multivariate information dynamics in networks», *Journal of Open Source Software*, 4(34):1081, 2019.
- A. Kraskov, H. Stögbauer y P. Grassberger, «Estimating mutual information», *Physical Review E*, 69:066138, 2004.

Los datos del sistema electromotor del pez de pulso eléctrico proceden del repositorio [Nexus](https://github.com/angellareo/nexus) de Ángel Lareo, asociado al modelo de Lareo, Varona y Rodríguez (2022).
