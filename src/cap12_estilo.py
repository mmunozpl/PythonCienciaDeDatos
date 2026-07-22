"""Estilo comun de las figuras del cap. 12 (paleta Okabe-Ito, marcas
finas, ejes recesivos). Todas las figuras del capitulo lo importan para
leerse como un sistema coherente y ser aptas para daltonicos."""
from __future__ import annotations

import matplotlib as mpl

# paleta Okabe-Ito: 8 colores disenados para daltonismo (Color Universal
# Design, Okabe & Ito 2008). Orden fijo, nunca ciclado.
OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73",
             "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
AZUL, NARANJA, VERDE = "#0072B2", "#E69F00", "#009E73"
BERMELLON, CIELO, MORADO = "#D55E00", "#56B4E9", "#CC79A7"
TINTA, GRIS = "#1a1a1a", "#8a8a8a"


def aplicar() -> None:
    """Fija los rcParams del libro: ejes recesivos, sin marco superior/
    derecho, tipografia serif (como el PDF), ciclo Okabe-Ito."""
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 120,
        "font.family": "serif",
        "font.size": 10,
        "axes.prop_cycle": mpl.cycler(color=OKABE_ITO[1:]),  # sin el negro
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": GRIS,
        "axes.labelcolor": TINTA,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": TINTA,
        "ytick.color": TINTA,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "lines.linewidth": 2.0,
    })
