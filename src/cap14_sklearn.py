"""Código reproducible del cap. 14 «scikit-learn y el aprendizaje moderno».

Trabaja sobre el catálogo de música (`data/processed/musica.parquet`, Spotify
Tracks, 113 999 pistas reales con rasgos de audio). La tarea de regresión
predice el volumen `loudness` (dB) a partir de los rasgos de audio; a
diferencia del cap. 13, aquí SÍ usamos una categórica (`track_genre`, más
`explicit`) mediante un ColumnTransformer, lo que aporta señal y baja el error.
La tarea de clasificación adivina el género por el sonido sobre un puñado de
seis géneros equilibrados (1000 pistas cada uno).

Para que el capítulo se reproduzca en segundos, la regresión comparte tablero
con la clasificación: el puñado de seis géneros (6000 pistas, 1000 por género,
`random_state=2026`). El patrón ---pipeline, validación, despliegue--- escala
igual a las 114 000. El paralelismo se fija en `n_jobs=4` (nunca -1) para no
sobresuscribir la CPU.

Genera las cinco figuras de datos del capítulo en `latex/figures/`:
    cap14_cv.pdf  cap14_importancia.pdf  cap14_grid.pdf  cap14_pdp.pdf
    cap14_roc.pdf

Uso:
    python src/cap14_sklearn.py --cifras     # imprime las cifras del texto
    python src/cap14_sklearn.py --figuras    # regenera las cinco figuras
    python src/cap14_sklearn.py              # ambas

Entorno verificado: scikit-learn 1.7.2, numpy 2.3.3, pandas 2.3.3 (CPython 3.11).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import (GridSearchCV, KFold, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.simplefilter("ignore")

RAIZ = Path(__file__).resolve().parents[1]
PARQUET = RAIZ / "data" / "processed" / "musica.parquet"
FIG = RAIZ / "latex" / "figures"
SEMILLA = 2026
N_JOBS = 4          # paralelismo acotado: nunca -1 (sobresuscribe la CPU)

# regresion: predecir el volumen desde rasgos de audio (numericos) y el genero
NUM = ["energy", "acousticness", "danceability", "valence"]
CAT = ["track_genre", "explicit"]

# clasificacion: adivinar el genero por el sonido (los diez rasgos de audio)
FEATS10 = ["danceability", "energy", "loudness", "speechiness", "acousticness",
           "instrumentalness", "liveness", "valence", "tempo", "duration_ms"]
PUNADO = ["pop", "rock", "classical", "hip-hop", "jazz", "reggaeton"]


def carga():
    """Devuelve (X_train, X_test, y_train, y_test) con partición reproducible.

    La regresión usa el mismo puñado de seis géneros que la clasificación.
    """
    df = _subconjunto_genero()
    X = df[NUM + CAT]
    y = df["loudness"].to_numpy(float)
    return train_test_split(X, y, test_size=0.2, random_state=SEMILLA)


def preprocesado():
    """El ColumnTransformer: estandariza lo numérico, codifica lo categórico.

    Con 114 géneros la codificación one-hot es ancha, así que pedimos salida
    densa (`sparse_output=False`) para que los estimadores la acepten sin más.
    """
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore",
                              sparse_output=False), CAT),
    ])


def pipeline_bosque():
    """El modelo protagonista del capítulo: el bosque aleatorio."""
    return Pipeline([
        ("prep", preprocesado()),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=SEMILLA,
                                     n_jobs=N_JOBS)),
    ])


def cifras():
    """Imprime las cifras que aparecen en el texto (verificación por ejecución)."""
    Xtr, Xte, ytr, yte = carga()
    print("formas:", Xtr.shape, Xte.shape)
    kf = KFold(5, shuffle=True, random_state=SEMILLA)

    # 14.1: modelo lineal y arbol sobre solo las numericas
    lin = LinearRegression().fit(Xtr[NUM], ytr)
    print(f"[14.1] lineal solo-num  R2={lin.score(Xte[NUM], yte):.2f}"
          f"  coef={lin.coef_.round(2)}")
    arb = DecisionTreeRegressor(max_depth=5, random_state=SEMILLA).fit(Xtr[NUM], ytr)
    print(f"[14.1] arbol max_depth=5 R2={arb.score(Xte[NUM], yte):.2f}")

    pipe = pipeline_bosque()
    pipe.fit(Xtr, ytr)
    print(f"[14.2] pipeline num+cat  MAE={mean_absolute_error(yte, pipe.predict(Xte)):.2f}"
          f"  R2={r2_score(yte, pipe.predict(Xte)):.3f}")
    # el modelo equivalente sin el genero (solo rasgos de audio), como el cap. 13
    p4 = Pipeline([("sc", StandardScaler()),
                   ("rf", RandomForestRegressor(n_estimators=100,
                                                random_state=SEMILLA, n_jobs=N_JOBS))])
    p4.fit(Xtr[NUM], ytr)
    print(f"[14.2] pipeline num-only MAE={mean_absolute_error(yte, p4.predict(Xte[NUM])):.2f}"
          f"  R2={r2_score(yte, p4.predict(Xte[NUM])):.3f}")

    print("[14.3] validacion cruzada (5 pliegues):")
    modelos = [
        ("lineal", Pipeline([("prep", preprocesado()), ("m", LinearRegression())])),
        ("boosting", Pipeline([("prep", preprocesado()),
                               ("m", HistGradientBoostingRegressor(
                                   random_state=SEMILLA))])),
        ("bosque", pipe),
    ]
    for nombre, est in modelos:
        sc = -cross_val_score(est, Xtr, ytr, cv=kf,
                              scoring="neg_mean_absolute_error", n_jobs=N_JOBS)
        print(f"       {nombre:>10}: MAE = {sc.mean():.2f} +/- {sc.std():.2f}")

    print("[14.3] GridSearchCV (bosque):")
    malla = {"rf__max_features": [0.3, 0.6, 1.0],
             "rf__min_samples_leaf": [1, 2, 5]}
    gs = GridSearchCV(pipe, malla, cv=kf, scoring="neg_mean_absolute_error",
                      n_jobs=N_JOBS)
    gs.fit(Xtr, ytr)
    print("       mejor:", gs.best_params_, "| MAE CV:", round(-gs.best_score_, 2))
    print(f"       en test: MAE={mean_absolute_error(yte, gs.predict(Xte)):.2f}")

    print("[14.4] arbol unico sin podar (CV):")
    t = -cross_val_score(Pipeline([("prep", preprocesado()),
                                   ("t", DecisionTreeRegressor(random_state=SEMILLA))]),
                         Xtr, ytr, cv=kf, scoring="neg_mean_absolute_error",
                         n_jobs=N_JOBS)
    print(f"       MAE = {t.mean():.2f} +/- {t.std():.2f}")

    print("[14.4] importancia por impureza del bosque (top 6):")
    nombres = pipe.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(pipe.named_steps["rf"].feature_importances_, index=nombres)
    for nom, v in imp.sort_values(ascending=False).head(6).round(3).items():
        print(f"       {nom:>22}: {v}")
    gcols = [n for n in nombres if n.startswith("cat__track_genre")]
    print(f"       (genero repartido en {len(gcols)} columnas suma {imp[gcols].sum():.3f})")

    print("[14.4] boosting con categoricas nativas (from_dtype):")
    _boosting_nativo()

    print("[14.4] linea base (media de loudness por genero):")
    tr = Xtr.copy(); tr["loudness"] = ytr
    tabla = tr.groupby("track_genre")["loudness"].mean()
    base = Xte["track_genre"].map(tabla).fillna(ytr.mean()).to_numpy()
    print(f"       regla por genero MAE={mean_absolute_error(yte, base):.2f}")

    print("[14.5] importancia por permutacion (n_repeats=10):")
    r = permutation_importance(pipe, Xte, yte, n_repeats=10, random_state=SEMILLA,
                               scoring="neg_mean_absolute_error", n_jobs=N_JOBS)
    for nom, im in sorted(zip(NUM + CAT, r.importances_mean), key=lambda t: -t[1]):
        print(f"       {nom:>12}: {im:.2f}")

    _clasificacion()


def _boosting_nativo():
    """El boosting acepta las categóricas sin one-hot (dtype 'category')."""
    df = _subconjunto_genero().copy()
    for c in CAT:
        df[c] = df[c].astype("category")
    y = df["loudness"].to_numpy(float)
    Xtr, Xte, ytr, yte = train_test_split(df[NUM + CAT], y, test_size=0.2,
                                          random_state=SEMILLA)
    gb = HistGradientBoostingRegressor(categorical_features="from_dtype",
                                       random_state=SEMILLA).fit(Xtr, ytr)
    print(f"       MAE={mean_absolute_error(yte, gb.predict(Xte)):.2f}"
          f"  R2={r2_score(yte, gb.predict(Xte)):.3f}"
          f"  categoricas={int(gb.is_categorical_.sum())}")


def _subconjunto_genero():
    """El puñado de seis géneros, 1000 pistas por género (6000 en total)."""
    df = pd.read_parquet(PARQUET)
    partes = [df[df["track_genre"] == g].sample(1000, random_state=SEMILLA)
              for g in PUNADO]
    return pd.concat(partes).reset_index(drop=True)


def _clasificacion():
    """Cifras de la sección de clasificación: adivinar el género por el sonido."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_recall_fscore_support)
    from sklearn.model_selection import StratifiedKFold
    sub = _subconjunto_genero()
    X = sub[FEATS10]
    y = sub["track_genre"].to_numpy()
    print(f"[14.6] clasificacion  {len(sub)} pistas, {len(PUNADO)} generos"
          f"  (tasa base = {1 / len(PUNADO):.3f})")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=SEMILLA, stratify=y)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEMILLA)
    for nombre, clf in [("logistica", LogisticRegression(max_iter=1000)),
                        ("bosque", RandomForestClassifier(
                            n_estimators=300, random_state=SEMILLA, n_jobs=N_JOBS))]:
        pipe = Pipeline([("sc", StandardScaler()), ("clf", clf)])
        acc = cross_val_score(pipe, Xtr, ytr, cv=skf, scoring="accuracy",
                              n_jobs=N_JOBS).mean()
        f1 = cross_val_score(pipe, Xtr, ytr, cv=skf, scoring="f1_macro",
                             n_jobs=N_JOBS).mean()
        print(f"       {nombre:>10}: acc={acc:.3f} f1_macro={f1:.3f}")
    pipe = Pipeline([("sc", StandardScaler()),
                     ("clf", RandomForestClassifier(
                         n_estimators=300, random_state=SEMILLA,
                         n_jobs=N_JOBS))]).fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    print(f"       TEST bosque: acc={accuracy_score(yte, pred):.3f}"
          f" f1_macro={f1_score(yte, pred, average='macro'):.3f}")
    prec, rec, f1g, _ = precision_recall_fscore_support(yte, pred, labels=PUNADO)
    print("       rendimiento por genero (precision, sensibilidad, F1):")
    for g, p, r_, f in zip(PUNADO, prec, rec, f1g):
        print(f"         {g:>10}: {p:.2f} {r_:.2f} {f:.2f}")
    cm = confusion_matrix(yte, pred, labels=PUNADO)
    print("       matriz de confusion (filas=real, orden PUNADO):")
    for g, fila in zip(PUNADO, cm.tolist()):
        print(f"         {g:>10}: {fila}")
    r = permutation_importance(pipe, Xte, yte, n_repeats=10,
                               random_state=SEMILLA, scoring="accuracy",
                               n_jobs=N_JOBS)
    print("       importancia por permutacion (caida de la exactitud, top 5):")
    for nom, im in sorted(zip(FEATS10, r.importances_mean),
                          key=lambda t: -t[1])[:5]:
        print(f"         {nom:>16}: {im:.3f}")
    # caja honesta: la popularidad NO sale del sonido (correlaciones ~ 0)
    df = pd.read_parquet(PARQUET)
    rmax = df[FEATS10].corrwith(df["popularity"]).abs().max()
    print(f"[14.6c] popularidad vs sonido: |r| maximo = {rmax:.3f} (todos < 0,10)")


# --- Figuras (estilo Okabe-Ito del cap. 12) ------------------------------------

def _estilo():
    import matplotlib as mpl
    mpl.use("Agg")
    sys.path.insert(0, str(RAIZ / "src"))
    from cap12_estilo import AZUL, GRIS, aplicar  # noqa: E402
    aplicar()
    return AZUL, GRIS


def figuras():
    import matplotlib.pyplot as plt
    AZUL, GRIS = _estilo()
    Xtr, Xte, ytr, yte = carga()
    prep = preprocesado()
    pipe = pipeline_bosque()
    kf = KFold(5, shuffle=True, random_state=SEMILLA)
    FIG.mkdir(parents=True, exist_ok=True)

    # cap14_cv: MAE por CV de tres modelos (el bosque gana)
    nombres, medias, desv = [], [], []
    for n, m in [("lineal", LinearRegression()),
                 ("boosting", HistGradientBoostingRegressor(random_state=SEMILLA)),
                 ("random\nforest", RandomForestRegressor(100, random_state=SEMILLA,
                                                          n_jobs=N_JOBS))]:
        sc = -cross_val_score(Pipeline([("prep", prep), ("m", m)]), Xtr, ytr,
                              cv=kf, scoring="neg_mean_absolute_error", n_jobs=N_JOBS)
        nombres.append(n); medias.append(sc.mean()); desv.append(sc.std())
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.bar(nombres, medias, yerr=desv, color=[GRIS, AZUL, AZUL], width=0.6, capsize=5)
    for i, (mm, dd) in enumerate(zip(medias, desv)):
        ax.text(i, mm + dd + 0.03, f"{mm:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("MAE por validación cruzada (dB)")
    ax.set_title("Tres modelos, un veredicto honesto")
    ax.set_ylim(0, max(medias) * 1.35)
    fig.tight_layout(); fig.savefig(FIG / "cap14_cv.pdf"); plt.close(fig)

    # cap14_importancia: permutación sobre el bosque
    pipe.fit(Xtr, ytr)
    r = permutation_importance(pipe, Xte, yte, n_repeats=10, random_state=SEMILLA,
                               scoring="neg_mean_absolute_error", n_jobs=N_JOBS)
    orden = np.argsort(r.importances_mean)
    nombres = np.array(NUM + CAT)[orden]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.barh(range(len(nombres)), r.importances_mean[orden],
            xerr=r.importances_std[orden], color=AZUL, capsize=3)
    ax.set_yticks(range(len(nombres))); ax.set_yticklabels(nombres)
    ax.set_xlabel("aumento del MAE al barajar (dB)")
    ax.set_title("Importancia por permutación")
    fig.tight_layout(); fig.savefig(FIG / "cap14_importancia.pdf"); plt.close(fig)

    # cap14_grid: mapa de calor de la búsqueda en malla (bosque)
    malla = {"rf__max_features": [0.3, 0.6, 1.0],
             "rf__min_samples_leaf": [1, 2, 5]}
    gs = GridSearchCV(pipe, malla, cv=kf, scoring="neg_mean_absolute_error",
                      n_jobs=N_JOBS)
    gs.fit(Xtr, ytr)
    res = -gs.cv_results_["mean_test_score"].reshape(3, 3)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    im = ax.imshow(res, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels([1, 2, 5])
    ax.set_yticks(range(3)); ax.set_yticklabels([0.3, 0.6, 1.0])
    ax.set_xlabel("min_samples_leaf"); ax.set_ylabel("max_features")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{res[i, j]:.2f}", ha="center", va="center",
                    color="white" if res[i, j] > res.min() + 0.08 else "black",
                    fontsize=9)
    ax.set_title("Búsqueda en malla (MAE por CV)")
    fig.colorbar(im, ax=ax, label="MAE (dB)", shrink=0.8)
    fig.tight_layout(); fig.savefig(FIG / "cap14_grid.pdf"); plt.close(fig)

    # cap14_pdp: dependencia parcial de dos rasgos de audio continuos.
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
    PartialDependenceDisplay.from_estimator(pipe, Xtr, ["energy", "acousticness"],
                                            ax=axs, line_kw={"color": AZUL})
    axs[0].set_title("Dependencia parcial: energy")
    axs[1].set_title("Dependencia parcial: acousticness")
    fig.suptitle("Cómo el modelo usa cada variable", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "cap14_pdp.pdf"); plt.close(fig)

    # cap14_roc: matriz de confusion del clasificador de genero (6 x 6).
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import ConfusionMatrixDisplay
    sub = _subconjunto_genero()
    Xc, yc = sub[FEATS10], sub["track_genre"].to_numpy()
    Xct, Xcte, yct, ycte = train_test_split(Xc, yc, test_size=0.2,
                                            random_state=SEMILLA, stratify=yc)
    clf = Pipeline([("sc", StandardScaler()),
                    ("clf", RandomForestClassifier(n_estimators=300,
                                                   random_state=SEMILLA,
                                                   n_jobs=N_JOBS))]).fit(Xct, yct)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ConfusionMatrixDisplay.from_estimator(
        clf, Xcte, ycte, labels=PUNADO, normalize="true", cmap="Blues",
        values_format=".2f", ax=ax, colorbar=False)
    ax.set_title("Adivinar el género por el sonido")
    ax.set_xlabel("género predicho"); ax.set_ylabel("género real")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    fig.tight_layout(); fig.savefig(FIG / "cap14_roc.pdf"); plt.close(fig)

    for n in ["cv", "importancia", "grid", "pdp", "roc"]:
        print("generada:", (FIG / f"cap14_{n}.pdf").name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Código del cap. 14 (scikit-learn).")
    ap.add_argument("--cifras", action="store_true")
    ap.add_argument("--figuras", action="store_true")
    a = ap.parse_args()
    if not (a.cifras or a.figuras):
        a.cifras = a.figuras = True
    if a.cifras:
        cifras()
    if a.figuras:
        figuras()
