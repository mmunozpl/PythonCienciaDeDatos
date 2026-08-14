# Python for Data Science

🇬🇧 English · 🇪🇸 [Español](LEEME.md)

**Manuel Muñoz Plá**

[![Cite](https://img.shields.io/badge/Cite-BibTeX-009e73)](#how-to-cite)

An introduction to **programming for data science**, brought up to 2026
state of the art (`uv`, `ruff`, `polars`, pandas' Arrow backend, modern
scikit-learn). The running case study is **music** (Spotify Tracks Dataset).
Sixteen chapters in five parts: *Language fundamentals*, *Data structures and
program design*, *The numeric and tabular stack*, *Data, statistics and
visualization*, and *Modeling, engineering and responsibility*.

This repository holds a **browsable web preview** of the book and the
**reproducible code** behind every result. Each chapter's exercises, their
solutions, and the appendices live in the **complete work** (print, PDF and
EPUB), distributed separately.

> 📘 **Book page** and more of the author's work: [manpla.net/libros/python-ciencia-datos](https://manpla.net/libros/python-ciencia-datos/)

## Contents

```
.
├── src/                 # one reproducible Python module per chapter
├── data/                # the data the code and figures consume
├── binder/              # reproducible environment for mybinder.org (runs in
│                        #   the cloud)
├── pyproject.toml       # dependencies — `uv sync` for your own machine
├── uv.lock              # exact versions (bit-for-bit reproducibility)
└── requirements.txt     # what Binder uses (mybinder.org needs it this way)
```

The web edition does not live here: it is read on manpla.net, linked below.

## Read the book

The web edition —16 chapters, with search, maths and per-page bibliography— is
published on the author's site:

> https://manpla.net/libros/python-ciencia-datos/

Each chapter is a page of manpla.net, with its own navigation, and is composed
with **vector SVG figures rendered with the same pdfLaTeX as the book**, so
references and figure/table/listing numbers match the printed text. This
repository holds the code and the data behind it.

Each chapter's full exercises, their solutions, and the appendices are not
published on the web: they live in the **complete work** (print, PDF and
EPUB).

## Run the code

Each chapter ships a reproducible module in `src/capNN_*.py`, seeded for
determinism: running it regenerates the figures and figures it relies on.

With [`uv`](https://docs.astral.sh/uv/) —the tool the book itself teaches
(chapters 1 and 15) for reproducible environments—:

```bash
uv sync                           # exact uv.lock environment, nothing to activate
uv run python src/cap07_numpy.py  # e.g. vectorized computation with NumPy
```

Without `uv`, with plain `pip`:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 src/cap07_numpy.py
```

Every chapter on the web edition also has a **"Run this chapter in Binder"**
button that opens JupyterLab in the cloud, with the environment and data
already set up — nothing to install.

## License

© 2026 **Manuel Muñoz Plá**.

| Part | What it is | License |
|---|---|---|
| `src/`, `data/`, `binder/` | Reproducible code, data and infrastructure | [MIT](src/LICENSE) — free to use |
| Book text (web edition on manpla.net) | The work itself | [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) — read and share with attribution; no commercial use or derivative works |

The **complete work** —with each chapter's exercises, their solutions, and the
appendices— is published in print, PDF and EPUB, all rights reserved.

The text carrying a restrictive license does **not limit use of the code**:
you're free to take the modules in `src/` or the environment in `binder/`
into a project of your own, commercial included, under the terms of the MIT
license.

The music data in `data/` comes from the *Spotify Tracks Dataset* by
maharshipandya (Hugging Face), published under the BSD license (with CC0
mirrors), and is not covered by this work's license.

## How to cite

If you mention or use this work, please cite it like this (GitHub also offers
a "Cite this repository" button, generated from `CITATION.cff`):

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

*Author: Manuel Muñoz Plá.*
