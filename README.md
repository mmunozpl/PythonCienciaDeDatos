# Python para la Ciencia de Datos

**Manuel Muñoz Plá**

Manual de introducción a la **programación para la ciencia de datos**, del nivel
principiante al universitario avanzado, actualizado al estado del arte de 2026
(`uv`, `ruff`, `polars`, backend Arrow de pandas, scikit-learn moderno).
El caso transversal es la **música** (Spotify Tracks Dataset). Dieciséis
capítulos en cinco partes: *Fundamentos del lenguaje*, *Estructuras de datos y
diseño de programas*, *El stack numérico y tabular*, *Datos, estadística y
visualización* y *Modelado, ingeniería y responsabilidad*.

Este repositorio reúne una **vista previa web navegable** del libro y el
**código reproducible** que genera cada resultado. Los ejercicios de cada
capítulo, sus soluciones y los apéndices están en la **obra completa** (papel,
PDF y EPUB), que se distribuye por separado.

## Contenido

```
.
├── docs/                # edición web (Quarto): un HTML por capítulo, figuras SVG,
│                        #   buscador y matemáticas
├── src/                 # un módulo Python reproducible por capítulo
├── data/                # los datos que el código y las figuras consumen
├── binder/              # entorno reproducible para mybinder.org (ejecutar en la nube)
└── requirements.txt
```

## Leer el libro

La edición web (16 capítulos, con buscador, matemáticas y bibliografía por
página) se publica con GitHub Pages desde `docs/`:

> https://mmunozpl.github.io/PythonCienciaDeDatos/

Cada capítulo se compone con **figuras vectoriales SVG renderizadas con el mismo
pdfLaTeX del libro**, de modo que las referencias y los números de figura, tabla
y listado son los del texto impreso.

Los amplios ejercicios de cada capítulo, sus soluciones y los apéndices no se
publican en la web: viven en la **obra completa** (papel, PDF y EPUB).

## Ejecutar el código

Cada capítulo trae un módulo reproducible en `src/capNN_*.py`, determinista con
semilla fija: al ejecutarlo regenera las cifras y figuras sobre las que se apoya.

```bash
python3 -m venv .venv && source .venv/bin/activate   # o conda
pip install -r requirements.txt
python3 src/cap07_numpy.py        # p. ej. cómputo vectorizado con NumPy
```

Además, cada capítulo de la web trae un botón **«Ejecutar este capítulo en
Binder»** que abre un JupyterLab en la nube, con el entorno y los datos ya
listos, sin instalar nada.

## Licencia

© 2026 **Manuel Muñoz Plá**.

| Parte | Qué es | Licencia |
|---|---|---|
| `src/`, `data/`, `binder/` | Código reproducible, datos e infraestructura | [MIT](src/LICENSE) — uso libre |
| `docs/` | Texto del libro (edición web) | [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.es) — leer y compartir con atribución; sin uso comercial ni obras derivadas |

La **obra completa** —con los ejercicios de cada capítulo, sus soluciones y los
apéndices— se publica en papel, PDF y EPUB con todos los derechos reservados.

Que el texto lleve una licencia restrictiva **no limita el uso del código**:
puedes llevarte los módulos de `src/` o el entorno de `binder/` a un proyecto
propio, incluso comercial, en los términos de la licencia MIT.

Los datos de música de `data/` proceden del *Spotify Tracks Dataset* de
maharshipandya (Hugging Face), publicado bajo licencia BSD (con espejos CC0), y
no están cubiertos por la licencia de esta obra.

## Cómo citar

Si mencionas o usas esta obra, cítala así (GitHub también ofrece el botón
«Cite this repository», generado desde `CITATION.cff`):

```bibtex
@book{munozpla2026pythoncienciadedatos,
  author    = {Muñoz Plá, Manuel},
  title     = {Python para la Ciencia de Datos},
  year      = {2026},
  publisher = {qWORD.dev},
  url       = {https://mmunozpl.github.io/PythonCienciaDeDatos/}
}
```

---

*Autor: Manuel Muñoz Plá.*
