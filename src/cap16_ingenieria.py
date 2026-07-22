"""Código reproducible del cap. 16 (ingeniería, reproducibilidad y ética).

Reúne las piezas EJECUTABLES del capítulo, todas sobre los datos del libro:
  - `normalizar`: función de referencia con anotaciones de tipo, sus pruebas al
    estilo `pytest` y una prueba BASADA EN PROPIEDADES con `hypothesis`.
  - `esquema_musica`: un esquema `pandera` que valida los rasgos de audio.
  - `deriva`: detección de deriva de datos (KS y PSI) entre dos géneros.
  - `auditoria_equidad`: auditoría de sesgo del clasificador de `premium` del
    cap. 15 (perfiles de escucha sintéticos) por sexo.

Uso:
    python src/cap16_ingenieria.py --cifras     # imprime las cifras del texto
    python src/cap16_ingenieria.py --figuras    # regenera las figuras
    pytest src/cap16_ingenieria.py              # ejecuta las pruebas
    ruff check src/cap16_ingenieria.py          # analiza el estilo

Entorno verificado: scikit-learn 1.7.2, pandera 0.32.1, scipy 1.17.1,
hypothesis 6.155.7, ruff 0.15.20, pytest 9.0.3 (CPython 3.11).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

RAIZ = Path(__file__).resolve().parents[1]
FIG = RAIZ / "latex" / "figures"
MUSICA = RAIZ / "data" / "processed" / "musica.parquet"
PERFILES = RAIZ / "data" / "processed" / "perfiles_escucha.parquet"
SEMILLA = 2026


# --- 16.2 Pruebas: función de referencia, por ejemplo y por propiedad --------

def normalizar(valores: Sequence[float]) -> list[float]:
    """Escala a [0, 1]; si todos los valores son iguales, ceros."""
    if not valores:
        raise ValueError("no se puede normalizar una secuencia vacía")
    lo, hi = min(valores), max(valores)
    if hi == lo:
        return [0.0 for _ in valores]
    return [(v - lo) / (hi - lo) for v in valores]


def test_normalizar_rango() -> None:
    assert normalizar([0, 5, 10]) == [0.0, 0.5, 1.0]


def test_normalizar_constante() -> None:
    assert normalizar([7, 7, 7]) == [0.0, 0.0, 0.0]  # caso límite explícito


def test_normalizar_vacia_avisa() -> None:
    import pytest
    with pytest.raises(ValueError):
        normalizar([])


def prueba_propiedad_en_rango() -> str:
    try:
        from hypothesis import given
        from hypothesis import strategies as st
    except ModuleNotFoundError:
        return "hypothesis no disponible: se omite la prueba de propiedad"

    @given(st.lists(st.floats(allow_nan=False, allow_infinity=False,
                              min_value=-1e6, max_value=1e6), min_size=1))
    def propiedad(xs: list[float]) -> None:
        assert all(0.0 <= r <= 1.0 for r in normalizar(xs))  # invariante

    propiedad()
    return "propiedad verificada con cientos de entradas generadas"


# --- 16.4 Validación de datos con pandera ------------------------------------

def esquema_musica():
    """Un esquema que declara qué es un rasgo de audio VÁLIDO."""
    import pandera.pandas as pa
    return pa.DataFrameSchema({
        "danceability": pa.Column(float, pa.Check.in_range(0, 1)),
        "energy": pa.Column(float, pa.Check.in_range(0, 1)),
        "tempo": pa.Column(float, pa.Check.in_range(0, 250)),
        "valence": pa.Column(float, pa.Check.in_range(0, 1)),
    })


def valida_musica() -> None:
    df = pd.read_parquet(MUSICA)
    esquema = esquema_musica()
    esquema.validate(df[["danceability", "energy", "tempo", "valence"]])
    print("       pandera: los rasgos de audio cumplen el esquema (validado)")
    # un dato corrupto (energy imposible) debe ser RECHAZADO
    malo = df.head(3).copy()
    malo.loc[malo.index[0], "energy"] = -5.0
    try:
        esquema.validate(malo[["danceability", "energy", "tempo", "valence"]])
        print("       ERROR: deberia haber fallado")
    except Exception:
        print("       pandera: rechaza un energy negativo, como debe")


# --- 16.6 Deriva de datos (data drift) ---------------------------------------

def _psi(base: np.ndarray, nuevo: np.ndarray, bins: int = 10) -> float:
    """Índice de estabilidad de la población entre dos muestras."""
    cortes = np.quantile(base, np.linspace(0, 1, bins + 1))
    cortes[0] -= 1e-6
    cortes[-1] += 1e-6
    b = np.clip(np.histogram(base, bins=cortes)[0] / len(base), 1e-6, None)
    n = np.clip(np.histogram(nuevo, bins=cortes)[0] / len(nuevo), 1e-6, None)
    return float(np.sum((n - b) * np.log(n / b)))


def deriva() -> None:
    from scipy.stats import ks_2samp
    m = pd.read_parquet(MUSICA)
    ref = m[m["track_genre"] == "classical"]["acousticness"].to_numpy()
    nue = m[m["track_genre"] == "jazz"]["acousticness"].to_numpy()
    ks = ks_2samp(ref, nue)
    print("       deriva de acousticness, classical -> jazz (musica):")
    print(f"         media classical={ref.mean():.2f}  jazz={nue.mean():.2f}")
    print(f"         KS={ks.statistic:.3f} (p={ks.pvalue:.1e})  "
          f"PSI={_psi(ref, nue):.2f}")


# --- 16.7 Auditoría de equidad (fairness) sobre los perfiles del cap. 15 ------

def _modelo_premium():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    df = pd.read_parquet(PERFILES)
    X = df[["edad", "minutos_dia", "energia_media", "n_artistas"]].copy()
    X["sexo_F"] = (df["sexo"] == "F").astype(int)
    y = df["premium"].to_numpy()
    Xtr, Xte, ytr, yte, _, ste = train_test_split(
        X, y, df["sexo"].to_numpy(), test_size=0.3, random_state=SEMILLA,
        stratify=y)
    # premium se genera con un modelo logístico: la regresión logística lo
    # recupera limpiamente (rasgo bien especificado)
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    tau = float(np.quantile(proba, 1 - yte.mean()))   # selección ~ tasa base
    return yte, proba, ste, tau


def _metricas_grupo(yte, pred, mask):
    sel = pred[mask].mean()
    pos = yte[mask] == 1
    neg = yte[mask] == 0
    tpr = pred[mask][pos].mean() if pos.sum() else 0.0
    fpr = pred[mask][neg].mean() if neg.sum() else 0.0
    return sel, tpr, fpr


def auditoria_equidad() -> None:
    yte, proba, ste, tau = _modelo_premium()
    pred = (proba >= tau).astype(int)
    print(f"       auditoria de equidad por sexo (umbral={tau:.3f}):")
    for g in ["M", "F"]:
        sel, tpr, fpr = _metricas_grupo(yte, pred, ste == g)
        print(f"         sexo={g}: seleccion={sel:.3f} TPR={tpr:.3f} "
              f"FPR={fpr:.3f} (premium real={yte[ste == g].mean():.3f})")


def mitigacion_equidad() -> None:
    """Post-procesado: un umbral POR GRUPO iguala la tasa de verdaderos
    positivos (igualdad de oportunidades), usando el sexo al decidir."""
    yte, proba, ste, tau = _modelo_premium()
    objetivo = 0.43   # TPR comun al que llevamos ambos grupos

    def umbral(mask):
        positivos = proba[mask][yte[mask] == 1]
        return float(np.quantile(positivos, 1 - objetivo))

    tM, tF = umbral(ste == "M"), umbral(ste == "F")
    pred = np.where(ste == "M", proba >= tM, proba >= tF).astype(int)
    print(f"       mitigacion: umbral por grupo (M={tM:.3f}, F={tF:.3f}):")
    for g in ["M", "F"]:
        _, tpr, _ = _metricas_grupo(yte, pred, ste == g)
        print(f"         sexo={g}: TPR={tpr:.3f}")


def cifras() -> None:
    print("[16.2] pruebas de la funcion de referencia:")
    test_normalizar_rango()
    test_normalizar_constante()
    print("       pruebas por ejemplo: OK")
    print("      ", prueba_propiedad_en_rango())
    print("[16.4] validacion de datos:")
    valida_musica()
    print("[16.6] deriva de datos:")
    deriva()
    print("[16.7] etica y equidad:")
    auditoria_equidad()
    mitigacion_equidad()


def _guardar(fig, nombre: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / nombre)
    import matplotlib.pyplot as plt
    plt.close(fig)


def figuras() -> None:
    import matplotlib as mpl
    mpl.use("Agg")
    sys.path.insert(0, str(RAIZ / "src"))
    from cap12_estilo import AZUL, BERMELLON, aplicar  # noqa: E402
    aplicar()
    import matplotlib.pyplot as plt
    from scipy.stats import ks_2samp
    FIG.mkdir(parents=True, exist_ok=True)

    # cap16_drift: acousticness de dos generos (deriva sin eje temporal)
    m = pd.read_parquet(MUSICA)
    ref = m[m["track_genre"] == "classical"]["acousticness"].to_numpy()
    nue = m[m["track_genre"] == "jazz"]["acousticness"].to_numpy()
    ks = ks_2samp(ref, nue)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    binss = np.linspace(0, 1, 40)
    ax.hist(ref, bins=binss, density=True, color=AZUL, alpha=0.75,
            label=f"classical (media {ref.mean():.2f})")
    ax.hist(nue, bins=binss, density=True, color=BERMELLON, alpha=0.7,
            label=f"jazz (media {nue.mean():.2f})")
    ax.set_xlabel("acousticness")
    ax.set_ylabel("densidad")
    ax.set_title(f"Deriva: KS={ks.statistic:.2f}, PSI={_psi(ref, nue):.1f}")
    ax.legend()
    _guardar(fig, "cap16_drift.pdf")

    # cap16_fairness: seleccion / TPR / FPR por sexo (auditoria de equidad)
    yte, proba, ste, tau = _modelo_premium()
    pred = (proba >= tau).astype(int)
    grupos = ["M", "F"]
    metr = {g: _metricas_grupo(yte, pred, ste == g) for g in grupos}
    etiquetas = ["selección", "TPR\n(recall)", "FPR"]
    x = np.arange(len(etiquetas))
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.bar(x - 0.2, metr["M"], width=0.38, color=AZUL, label="hombres")
    ax.bar(x + 0.2, metr["F"], width=0.38, color=BERMELLON, label="mujeres")
    for g in grupos:
        desplaza = 0.2 if g == "F" else -0.2
        for j, v in enumerate(metr[g]):
            ax.text(j + desplaza, v + 0.005, f"{v:.2f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas)
    ax.set_ylabel("tasa")
    ax.set_title("Auditoría de equidad: las tasas no coinciden")
    ax.legend()
    _guardar(fig, "cap16_fairness.pdf")

    for n in ["drift", "fairness"]:
        print("generada:", (FIG / f"cap16_{n}.pdf").name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Código del cap. 16 (ingeniería).")
    ap.add_argument("--cifras", action="store_true")
    ap.add_argument("--figuras", action="store_true")
    a = ap.parse_args()
    if not (a.cifras or a.figuras):
        a.cifras = a.figuras = True
    if a.cifras:
        cifras()
    if a.figuras:
        figuras()
