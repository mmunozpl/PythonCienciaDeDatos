# Python para la Ciencia de Datos

🇬🇧 [English](README.md) · 🇪🇸 Español

**Manuel Muñoz Plá**

[![Cite](https://img.shields.io/badge/Cite-BibTeX-009e73)](#cómo-citar)

Manual de introducción a la **programación para la ciencia de datos**,
actualizado al estado del arte de 2026 (`uv`, `ruff`, `polars`, backend
Arrow de pandas, scikit-learn moderno). El caso transversal es la **música**
(Spotify Tracks Dataset). Dieciséis
capítulos en cinco partes: *Fundamentos del lenguaje*, *Estructuras de datos y
diseño de programas*, *El stack numérico y tabular*, *Datos, estadística y
visualización* y *Modelado, ingeniería y responsabilidad*.

Este repositorio reúne una **vista previa web navegable** del libro y el
**código reproducible** que genera cada resultado. Los ejercicios de cada
capítulo, sus soluciones y los apéndices están en la **obra completa** (papel,
PDF y EPUB), que se distribuye por separado.

> 📘 **Ficha del libro** y más obras del autor: [manpla.net/libros/python-ciencia-datos](https://manpla.net/libros/python-ciencia-datos/)

## Contenido

```
.
├── src/                 # un módulo Python reproducible por capítulo
├── data/                # los datos que el código y las figuras consumen
├── binder/              # entorno reproducible para mybinder.org (ejecutar en la nube)
├── pyproject.toml       # dependencias — `uv sync` para tu propio ordenador
├── uv.lock              # versiones exactas (reproducibilidad bit a bit)
└── requirements.txt     # el que usa Binder (mybinder.org lo pide así)
```

La edición web no vive aquí: se lee en manpla.net, enlazada más abajo.

## Leer el libro

La edición web —16 capítulos, con buscador, matemáticas y bibliografía por
página— se publica en el sitio del autor:

> https://manpla.net/libros/python-ciencia-datos/

Cada capítulo es una página de manpla.net, con su propia navegación, y se
compone con **figuras vectoriales SVG renderizadas con el mismo pdfLaTeX del
libro**, de modo que las referencias y los números de figura, tabla y listado
son los del texto impreso. Este repositorio guarda el código y los datos que
la respaldan.

Los amplios ejercicios de cada capítulo, sus soluciones y los apéndices no se
publican en la web: viven en la **obra completa** (papel, PDF y EPUB).

## Ejecutar el código

Cada capítulo trae un módulo reproducible en `src/capNN_*.py`, determinista con
semilla fija: al ejecutarlo regenera las cifras y figuras sobre las que se apoya.

Con [`uv`](https://docs.astral.sh/uv/) —la herramienta que enseña el propio
libro (cap. 1 y 15) para entornos reproducibles—:

```bash
uv sync                           # entorno exacto de uv.lock, sin activar nada
uv run python src/cap07_numpy.py  # p. ej. cómputo vectorizado con NumPy
```

Sin `uv`, con `pip` de toda la vida:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 src/cap07_numpy.py
```

Además, cada capítulo de la web trae un botón **«Ejecutar este capítulo en
Binder»** que abre un JupyterLab en la nube, con el entorno y los datos ya
listos, sin instalar nada.

## Licencia

© 2026 **Manuel Muñoz Plá**.

| Parte | Qué es | Licencia |
|---|---|---|
| `src/`, `data/`, `binder/` | Código reproducible, datos e infraestructura | [MIT](src/LICENSE) — uso libre |
| Texto del libro (edición web en manpla.net) | La obra | [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.es) — leer y compartir con atribución; sin uso comercial ni obras derivadas |

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
  url       = {https://manpla.net/libros/python-ciencia-datos/}
}
```

---

*Autor: Manuel Muñoz Plá.*
