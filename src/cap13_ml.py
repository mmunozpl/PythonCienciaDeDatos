"""Código del capítulo 13 — Fundamentos del aprendizaje automático.

Reúne el código verificado del capítulo sobre el catálogo de música
(`data/processed/musica.parquet`, Spotify Tracks: 113 999 pistas reales
con rasgos de audio). A diferencia de un dataset sintético, la música
TIENE señal real, así que no hace falta fabricarla. El capítulo trabaja
sobre un puñado equilibrado de seis
géneros (pop, rock, classical, hip-hop, jazz, reggaeton; 1000 pistas cada
uno, 6000 en total) y plantea dos tareas:

  * regresión: predecir el volumen `loudness` (dB) desde cuatro rasgos de
    audio, con la ecuación normal DESDE CERO y su equivalencia con
    scikit-learn;
  * clasificación: adivinar el `track_genre` por el sonido (los diez
    rasgos de audio), con la red neuronal de PyTorch como protagonista.

Demuestra: la partición train/test, la regresión lineal por ecuación
normal, la clasificación y sus métricas, la brecha de sobreajuste, el
descenso de gradiente y una red neuronal entrenada de verdad con PyTorch
(que clasifica el género). La red solo se entrena si torch está instalado
(degradación elegante).

Uso:
    python src/cap13_ml.py            # imprime las cifras del texto
    python src/cap13_ml.py --figuras  # regenera las cinco figuras de datos
    python src/cap13_ml.py --cifras --figuras   # ambas
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
PARQUET = RAIZ / "data" / "processed" / "musica.parquet"
FIG = RAIZ / "latex" / "figures"
SEMILLA = 2026

# el puñado de seis generos equilibrados (mismo subconjunto que el cap. 14)
PUNADO = ["pop", "rock", "classical", "hip-hop", "jazz", "reggaeton"]
# regresion: predecir el volumen desde cuatro rasgos de audio
RFEATS = ["energy", "acousticness", "danceability", "valence"]
# clasificacion: adivinar el genero desde los diez rasgos de audio
CFEATS = ["danceability", "energy", "loudness", "speechiness", "acousticness",
          "instrumentalness", "liveness", "valence", "tempo", "duration_ms"]


def subconjunto() -> pd.DataFrame:
    """El puñado de seis géneros, 1000 pistas por género (6000 en total).

    Se construye igual que en el cap. 14 para que las cifras concuerden.
    """
    df = pd.read_parquet(PARQUET)
    partes = [df[df["track_genre"] == g].sample(1000, random_state=SEMILLA)
              for g in PUNADO]
    return pd.concat(partes).reset_index(drop=True)


def ajustar_lineal(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Minimos cuadrados por ecuacion normal (con intercepto)."""
    Xb = np.c_[np.ones(len(X)), X]
    return np.linalg.solve(Xb.T @ Xb, Xb.T @ y)


def predecir_lineal(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.c_[np.ones(len(X)), X] @ beta


def descenso_gradiente(Xb: np.ndarray, y: np.ndarray, eta: float = 0.3,
                       pasos: int = 100) -> tuple[np.ndarray, list]:
    """Descenso de gradiente por lotes para minimos cuadrados. Xb ya trae
    la columna de unos y las caracteristicas estandarizadas."""
    beta = np.zeros(Xb.shape[1])
    n = len(Xb)
    perdidas = []
    for _ in range(pasos):
        err = Xb @ beta - y
        perdidas.append(float((err ** 2).mean()))
        beta -= eta * (2 / n) * (Xb.T @ err)
    return beta, perdidas


def red_pytorch(Xtr, ytr, Xte, yte, epocas: int = 100):
    """Entrena un MLP (10-32-16-6) que clasifica el genero con PyTorch.

    Devuelve (exactitud en test, curva de exactitud por epoca, predicciones)
    o None si torch no esta disponible (degradacion elegante).
    """
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError:
        return None
    torch.manual_seed(SEMILLA)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    yt = torch.tensor(ytr, dtype=torch.long, device=dev)
    Xe = torch.tensor(Xte, dtype=torch.float32, device=dev)
    ye = torch.tensor(yte, dtype=torch.long, device=dev)
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=256, shuffle=True)
    red = nn.Sequential(nn.Linear(10, 32), nn.ReLU(),
                        nn.Linear(32, 16), nn.ReLU(),
                        nn.Linear(16, 6)).to(dev)
    opt = torch.optim.Adam(red.parameters(), lr=0.01)
    lossf = nn.CrossEntropyLoss()
    curva = []
    for _ in range(epocas):
        red.train()
        for xb, yb in dl:
            opt.zero_grad()
            lossf(red(xb), yb).backward()
            opt.step()
        red.eval()
        with torch.no_grad():
            curva.append((red(Xe).argmax(1) == ye).float().mean().item())
    red.eval()
    with torch.no_grad():
        pred = red(Xe).argmax(1).cpu().numpy()
    return (float((pred == yte).mean()), curva, pred)


# --- Cifras (verificación por ejecución) -------------------------------------

def cifras() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import (LinearRegression, LogisticRegression,
                                      Lasso, Ridge)
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score,
                                 mean_absolute_error, r2_score, recall_score)
    from sklearn.model_selection import (KFold, cross_val_score,
                                         train_test_split)
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    sub = subconjunto()
    genero = sub["track_genre"].to_numpy()
    print(f"[13.1] subconjunto: {sub.shape} | {len(PUNADO)} generos "
          f"x 1000 pistas")
    print(f"       corr loudness~energy = "
          f"{sub['loudness'].corr(sub['energy']):.3f}")

    # ---------- Regresion: loudness desde rasgos de audio ----------
    Xr = sub[RFEATS].to_numpy(float)
    yr = sub["loudness"].to_numpy(float)
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        Xr, yr, test_size=0.2, random_state=SEMILLA, stratify=genero)
    print(f"[13.1] split: {Xr_tr.shape} {Xr_te.shape}")

    beta = ajustar_lineal(Xr_tr, yr_tr)
    lin = LinearRegression().fit(Xr_tr, yr_tr)
    igual = (np.allclose(lin.coef_, beta[1:])
             and np.allclose(lin.intercept_, beta[0]))
    print(f"[13.2] ecuacion normal: b={beta[0]:.2f} "
          f"coef={[round(float(v), 2) for v in beta[1:]]}")
    print(f"[13.2] == sklearn: {igual}")
    print(f"[13.2] MAE tr={mean_absolute_error(yr_tr, lin.predict(Xr_tr)):.2f} "
          f"te={mean_absolute_error(yr_te, lin.predict(Xr_te)):.2f} "
          f"R2={r2_score(yr_te, lin.predict(Xr_te)):.3f}")
    for n, c in zip(RFEATS, lin.coef_):
        print(f"       {n:14s} {c:+.2f}")

    print("[13.4] polinomios (train/test MAE):")
    for gr in (1, 2, 3, 5, 7, 9, 10):
        m = make_pipeline(PolynomialFeatures(gr),
                          LinearRegression()).fit(Xr_tr, yr_tr)
        print(f"       grado {gr:2d}: train "
              f"{mean_absolute_error(yr_tr, m.predict(Xr_tr)):.2f} "
              f"test {mean_absolute_error(yr_te, m.predict(Xr_te)):.2f}")
    ridge = Ridge(alpha=10.0).fit(Xr_tr, yr_tr)
    lasso = make_pipeline(StandardScaler(),
                          Lasso(alpha=0.8)).fit(Xr_tr, yr_tr)
    print(f"[13.4] Ridge test="
          f"{mean_absolute_error(yr_te, ridge.predict(Xr_te)):.2f} "
          f"Lasso test="
          f"{mean_absolute_error(yr_te, lasso.predict(Xr_te)):.2f}")
    print(f"[13.4] Lasso coef={lasso.named_steps['lasso'].coef_.round(2)}")
    kf = KFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    pts = cross_val_score(LinearRegression(), Xr, yr, cv=kf,
                          scoring="neg_mean_absolute_error")
    print(f"[13.4] CV MAE={(-pts).round(2)} media={-pts.mean():.2f} "
          f"desv={pts.std():.2f}")

    # ---------- Descenso de gradiente sobre la misma regresion ----------
    esc_r = StandardScaler().fit(Xr_tr)
    Xb = np.c_[np.ones(len(Xr_tr)), esc_r.transform(Xr_tr)]
    beta_gd, perd = descenso_gradiente(Xb, yr_tr, eta=0.3, pasos=100)
    beta_norm = np.linalg.solve(Xb.T @ Xb, Xb.T @ yr_tr)
    print(f"[13.5] GD perdida {perd[0]:.2f} -> {perd[-1]:.2f} "
          f"(iter 1={perd[1]:.2f}, 5={perd[5]:.2f}, 20={perd[20]:.2f})")
    print(f"[13.5] GD == ecuacion normal: "
          f"{np.allclose(beta_gd, beta_norm, atol=1e-3)}")

    # ---------- Clasificacion del genero ----------
    Xc = sub[CFEATS].to_numpy(float)
    labels = sorted(PUNADO)
    yc = np.array([labels.index(v) for v in genero])
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
        Xc, yc, test_size=0.2, random_state=SEMILLA, stratify=yc)
    esc_c = StandardScaler().fit(Xc_tr)
    Ztr, Zte = esc_c.transform(Xc_tr), esc_c.transform(Xc_te)

    logi = LogisticRegression(max_iter=1000).fit(Ztr, yc_tr)
    pl = logi.predict(Zte)
    trivial = Counter(yc_tr).most_common(1)[0][0]
    triv = np.full_like(yc_te, trivial)
    print(f"[13.3] logistica acc={accuracy_score(yc_te, pl):.3f} "
          f"macroF1={f1_score(yc_te, pl, average='macro'):.3f}")
    print(f"[13.3] trivial (siempre '{labels[trivial]}') "
          f"acc={accuracy_score(yc_te, triv):.3f}")
    print(f"[13.3] confusion (orden {labels}):")
    print(confusion_matrix(yc_te, pl))
    print(classification_report(yc_te, pl, target_names=labels, digits=2))

    hgb = HistGradientBoostingClassifier(random_state=SEMILLA).fit(Xc_tr, yc_tr)
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=800,
                        random_state=SEMILLA).fit(Ztr, yc_tr)
    print(f"[13.8] boosting acc={accuracy_score(yc_te, hgb.predict(Xc_te)):.3f} "
          f"| MLP sklearn acc={accuracy_score(yc_te, mlp.predict(Zte)):.3f}")

    red = red_pytorch(Ztr.astype(np.float32), yc_tr,
                      Zte.astype(np.float32), yc_te)
    if red is None:
        print("[13.7] (torch no disponible: se omite la red)")
    else:
        acc_red, curva, pred = red
        print(f"[13.7] red PyTorch acc={acc_red:.3f} "
              f"(epoca 25={curva[24]:.3f}, 50={curva[49]:.3f}, "
              f"75={curva[74]:.3f}, 100={curva[99]:.3f})")
        print(f"[13.7] red PyTorch macroF1="
              f"{f1_score(yc_te, pred, average='macro'):.3f}")

    # ---------- Caja honesta: la popularidad NO se predice del sonido ----------
    df = pd.read_parquet(PARQUET)
    r_max = df[CFEATS].corrwith(df["popularity"]).abs().max()
    print(f"[13.3c] popularidad ~ sonido: |r| max = {r_max:.3f} "
          f"(no se predice del sonido)")


# --- Figuras ------------------------------------------------------------------

def _estilo():
    import matplotlib as mpl
    mpl.use("Agg")
    import sys
    sys.path.insert(0, str(RAIZ / "src"))
    from cap12_estilo import AZUL, BERMELLON, GRIS, aplicar  # noqa: E402
    aplicar()
    return AZUL, BERMELLON, GRIS


def figuras() -> None:
    import matplotlib.pyplot as plt
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    AZUL, BERMELLON, GRIS = _estilo()
    FIG.mkdir(parents=True, exist_ok=True)
    sub = subconjunto()
    genero = sub["track_genre"].to_numpy()
    labels = sorted(PUNADO)

    # --- datos de regresion ---
    Xr = sub[RFEATS].to_numpy(float)
    yr = sub["loudness"].to_numpy(float)
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        Xr, yr, test_size=0.2, random_state=SEMILLA, stratify=genero)

    # --- datos de clasificacion ---
    Xc = sub[CFEATS].to_numpy(float)
    yc = np.array([labels.index(v) for v in genero])
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
        Xc, yc, test_size=0.2, random_state=SEMILLA, stratify=yc)
    esc = StandardScaler().fit(Xc_tr)
    Ztr, Zte = esc.transform(Xc_tr), esc.transform(Xc_te)

    # cap13_confusion: matriz de confusion 6x6 de la logistica
    logi = LogisticRegression(max_iter=1000).fit(Ztr, yc_tr)
    cm = confusion_matrix(yc_te, logi.predict(Zte))
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("género predicho"); ax.set_ylabel("género real")
    ax.set_title("¿Qué género es? (regresión logística)")
    umbral = cm.max() / 2
    for i in range(6):
        for j in range(6):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > umbral else "#1a1a1a",
                    fontsize=8)
    ax.grid(False)
    fig.tight_layout(); fig.savefig(FIG / "cap13_confusion.pdf")
    plt.close(fig)

    # cap13_sobreajuste: brecha train/test por grado del polinomio
    grados = list(range(1, 11))
    e_tr, e_te = [], []
    from sklearn.metrics import mean_absolute_error as mae
    for gr in grados:
        m = make_pipeline(PolynomialFeatures(gr),
                          LinearRegression()).fit(Xr_tr, yr_tr)
        e_tr.append(mae(yr_tr, m.predict(Xr_tr)))
        e_te.append(mae(yr_te, m.predict(Xr_te)))
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(grados, e_tr, "o-", color=AZUL, label="entrenamiento")
    ax.plot(grados, e_te, "s-", color=BERMELLON, label="test")
    ax.set_xlabel("grado del polinomio")
    ax.set_ylabel("MAE (dB)")
    ax.set_title("La brecha del sobreajuste")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "cap13_sobreajuste.pdf")
    plt.close(fig)

    # cap13_gradiente: la perdida cae con las iteraciones
    esc_r = StandardScaler().fit(Xr_tr)
    Xb = np.c_[np.ones(len(Xr_tr)), esc_r.transform(Xr_tr)]
    _b, perd = descenso_gradiente(Xb, yr_tr, eta=0.3, pasos=100)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(range(len(perd)), perd, color=AZUL)
    ax.set_xlabel("iteración")
    ax.set_ylabel("ECM (pérdida)")
    ax.set_title("El descenso de gradiente converge")
    fig.tight_layout(); fig.savefig(FIG / "cap13_gradiente.pdf")
    plt.close(fig)

    # cap13_curva_dl: exactitud de la red por epoca vs boosting
    hgb = HistGradientBoostingClassifier(random_state=SEMILLA).fit(Xc_tr, yc_tr)
    acc_hgb = accuracy_score(yc_te, hgb.predict(Xc_te))
    red = red_pytorch(Ztr.astype(np.float32), yc_tr,
                      Zte.astype(np.float32), yc_te)
    if red is not None:
        _acc, curva, _pred = red
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        ax.plot(range(1, len(curva) + 1), curva, color=AZUL,
                label="red PyTorch")
        ax.axhline(acc_hgb, ls=":", color=GRIS,
                   label=f"boosting ({acc_hgb:.2f})")
        ax.set_xlabel("época")
        ax.set_ylabel("exactitud en test")
        ax.set_title("La red aprende a distinguir géneros")
        ax.legend()
        fig.tight_layout(); fig.savefig(FIG / "cap13_curva_dl.pdf")
        plt.close(fig)

    # cap13_comparacion: exactitud de la escalera de modelos
    trivial = Counter(yc_tr).most_common(1)[0][0]
    triv = accuracy_score(yc_te, np.full_like(yc_te, trivial))
    acc_lin = accuracy_score(yc_te, logi.predict(Zte))
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=800,
                        random_state=SEMILLA).fit(Ztr, yc_tr)
    acc_mlp = accuracy_score(yc_te, mlp.predict(Zte))
    acc_red = red[0] if red is not None else np.nan
    nombres = ["trivial", "logística", "boosting", "MLP", "red\nPyTorch"]
    vals = [triv, acc_lin, acc_hgb, acc_mlp, acc_red]
    colores = [GRIS, GRIS, AZUL, AZUL, AZUL]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.bar(nombres, vals, color=colores, width=0.62)
    for i, v in enumerate(vals):
        if not np.isnan(v):
            ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("exactitud en test")
    ax.set_title("Cinco modelos, un veredicto")
    ax.set_ylim(0, 0.85)
    fig.tight_layout(); fig.savefig(FIG / "cap13_comparacion.pdf")
    plt.close(fig)

    for n in ("confusion", "sobreajuste", "gradiente", "curva_dl",
              "comparacion"):
        print("generada:", (FIG / f"cap13_{n}.pdf").name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cifras", action="store_true")
    ap.add_argument("--figuras", action="store_true")
    args = ap.parse_args()
    if not (args.cifras or args.figuras):
        args.cifras = True          # por defecto, solo las cifras
    if args.cifras:
        cifras()
    if args.figuras:
        figuras()
