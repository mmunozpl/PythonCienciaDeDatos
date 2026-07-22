"""Script maestro del cap. 12 «Visualizacion de datos».

Genera las 16 figuras del capitulo como PDF vectoriales, con estilo
coherente (paleta Okabe-Ito, apto para daltonicos) definido en
``cap12_estilo``. Cada figura ilustra una regla del metodo dataviz.

Ejecutar desde la raiz del proyecto (produce latex/figures/cap12_*.pdf):
    uv run python src/cap12_visualizacion.py
"""
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy import stats

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from cap12_estilo import (  # noqa: E402
    AZUL,
    BERMELLON,
    GRIS,
    NARANJA,
    TINTA,
    VERDE,
    aplicar,
)

aplicar()

DATA = Path("data/processed")
FIG = Path("latex/figures")

# el catalogo de musica (Spotify Tracks): 113 999 pistas, cero nulos.
# popularity (0-100) es el objetivo de analisis; los rasgos de audio
# (danceability, energy, valence...) son las magnitudes continuas y
# track_genre es la categoria para comparar.
DF = pd.read_parquet(DATA / "musica.parquet")

# rasgos [0, 1] comparables en un mismo eje (papel de NO2/PM10/O3)
FEAT3 = ["danceability", "energy", "valence"]
COLOR_FEAT = {"danceability": AZUL, "energy": NARANJA, "valence": VERDE}
# el «puñado» de generos reconocibles para comparar popularidad
PUNADO = ["classical", "jazz", "rock", "reggaeton", "hip-hop", "pop"]
# tres generos de firma acustica contrastada (agrupadas)
GEN3 = ["classical", "pop", "reggaeton"]
POP_MEDIA = float(DF["popularity"].mean())   # media global de popularidad


# --------------------------------------------------------------------
# utilidades
# --------------------------------------------------------------------
def guardar(fig, nombre, rect=None):
    """Ajusta, guarda como PDF vectorial determinista y cierra."""
    ruta = FIG / f"{nombre}.pdf"
    fig.tight_layout(rect=rect)
    fig.savefig(ruta, metadata={"CreationDate": None})
    plt.close(fig)
    return ruta.stat().st_size


def ecdf(valores):
    """Devuelve (x, y) de la distribucion acumulada empirica."""
    x = np.sort(valores)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def pop_por_genero(generos):
    """Popularidad media de cada genero del puñado, ordenada ascendente."""
    sel = DF[DF["track_genre"].isin(generos)]
    return sel.groupby("track_genre")["popularity"].mean().sort_values()


def perfil_loudness():
    """Loudness medio por nivel de popularidad (0-100): eje ordenado."""
    return DF.groupby("popularity")["loudness"].mean()


# --------------------------------------------------------------------
# figuras
# --------------------------------------------------------------------
def fig01_jerarquia(salida):
    """Jerarquia de Cleveland-McGill: posicion, longitud, angulo."""
    med = pop_por_genero(["pop", "emo", "anime", "grunge"])
    labels = list(med.index)
    vals = med.to_numpy()
    grises = ["#cfcfcf", "#a8a8a8", "#868686", "#5f5f5f"]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(9.6, 3.3))
    y = np.arange(len(labels))
    # (a) posicion sobre un eje comun
    a.hlines(y, 0, vals, color="#dddddd", lw=1)
    a.plot(vals, y, "o", color=AZUL, ms=9)
    a.set_yticks(y)
    a.set_yticklabels(labels)
    a.set_xlim(0, 60)
    a.set_xlabel("popularidad media")
    a.set_title("(a) Posición")
    # (b) longitud sin eje alineado
    despl = [3.0, 8.0, 1.0, 6.0]
    b.barh(y, vals, left=despl, color=AZUL, height=0.6)
    b.set_yticks(y)
    b.set_yticklabels(labels)
    b.set_xticks([])
    b.spines["bottom"].set_visible(False)
    b.grid(False)
    b.set_title("(b) Longitud")
    # (c) angulo
    c.pie(vals, labels=labels, colors=grises,
          wedgeprops={"edgecolor": "white", "linewidth": 1},
          textprops={"color": TINTA, "fontsize": 9})
    c.set_title("(c) Ángulo")
    fig.suptitle("La misma información, tres codificaciones",
                 fontsize=12, fontweight="bold")
    return guardar(fig, salida, rect=(0, 0, 1, 0.93))


def franjas_popularidad():
    """Reparto de pistas en cuatro franjas de popularidad."""
    cortes = [25.0, 50.0, 75.0]
    labs = ["baja", "media", "alta", "muy alta"]
    cat = pd.cut(DF["popularity"], [-1.0, *cortes, 101.0], labels=labs)
    prop = pd.Series(cat).value_counts(normalize=True)[labs] * 100
    return labs, prop.round(1)


def fig02_tarta_barras(salida):
    """Contraejemplo: la tarta pierde frente a barras ordenadas."""
    labs, prop = franjas_popularidad()
    grises = ["#cfcfcf", "#a8a8a8", "#828282", "#5f5f5f"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.7))
    a.pie(prop.to_numpy(), labels=labs, colors=grises,
          startangle=90, counterclock=False,
          autopct=lambda p: f"{p:.0f}%",
          wedgeprops={"edgecolor": "white", "linewidth": 1},
          textprops={"color": TINTA, "fontsize": 9})
    a.set_title("Tarta: comparar ángulos cuesta")
    orden = prop.sort_values(ascending=True)
    y = np.arange(len(orden))
    b.barh(y, orden.to_numpy(), color=AZUL, height=0.62)
    b.set_yticks(y)
    b.set_yticklabels(orden.index)
    b.set_xlabel("proporción de pistas (%)")
    b.set_xlim(0, orden.max() * 1.2)
    for yi, v in zip(y, orden.to_numpy(), strict=False):
        b.text(v + 0.8, yi, f"{v:.1f}%", va="center",
               color=TINTA, fontsize=9)
    b.set_title("Barras: orden y longitud directas")
    return guardar(fig, salida, rect=(0, 0, 1, 0.95))


def fig03_liefactor(salida):
    """El eje truncado como factor de mentira (lie factor)."""
    med = pop_por_genero(["pop", "emo"])
    labels = list(med.index)
    vals = med.to_numpy()
    base = float(vals[0]) * 0.98
    factor = vals[0] / (vals[0] - base)
    rel = (vals[1] - vals[0]) / vals[0] * 100
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.5, 3.8))
    x = [0, 1]
    a.bar(x, vals, color=BERMELLON, width=0.6)
    a.set_ylim(base, vals[1] + 0.3)
    a.set_xticks(x)
    a.set_xticklabels(labels)
    a.set_ylabel("popularidad media")
    a.set_title("Engañoso: eje desde 46,6")
    a.annotate(f"exagera la diferencia ≈{factor:.0f}×",
               xy=(0.5, 0.9), xycoords="axes fraction",
               ha="center", color=TINTA, fontsize=9)
    b.bar(x, vals, color=AZUL, width=0.6)
    b.set_ylim(0, 60)
    b.set_xticks(x)
    b.set_xticklabels(labels)
    b.set_ylabel("popularidad media")
    b.axhline(POP_MEDIA, color=GRIS, ls=":", lw=1)
    b.set_title("Honesto: eje desde 0")
    b.annotate(f"diferencia real: {rel:.1f}%",
               xy=(0.5, 0.5), xycoords="axes fraction",
               ha="center", color=TINTA, fontsize=9)
    return guardar(fig, salida, rect=(0, 0, 1, 0.95))


def fig04_anatomia(salida):
    """Anatomia de una figura matplotlib, elemento a elemento."""
    tempo = DF.loc[DF["tempo"] > 0, "tempo"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.hist(tempo, bins=24, color="#bcd6ea",
            edgecolor="white", linewidth=0.5)
    ax.set_title("Histograma de tempo  (title)")
    ax.set_xlabel("tempo (BPM)  (xlabel)")
    ax.set_ylabel("nº de pistas  (ylabel)")
    op = {"arrowstyle": "->", "color": TINTA, "lw": 1}

    def flecha(txt, xy, xytext, **kw):
        ax.annotate(txt, xy=xy, xytext=xytext, arrowprops=op,
                    color=TINTA, fontsize=8.5, **kw)

    flecha("Figure", (0.015, 0.02), (0.13, 0.11),
           xycoords="figure fraction", textcoords="figure fraction")
    flecha("Axes", (0.60, 0.55), (0.70, 0.74),
           xycoords="axes fraction", textcoords="axes fraction")
    flecha("spine", (0.0, 0.35), (0.15, 0.52),
           xycoords="axes fraction", textcoords="axes fraction")
    flecha("tick", (0.30, 0.0), (0.36, 0.15),
           xycoords="axes fraction", textcoords="axes fraction")
    return guardar(fig, salida)


def fig05_barras(salida):
    """Popularidad media por genero: barras ordenadas, la mayor en azul."""
    med = pop_por_genero(PUNADO).round(1)
    colores = [GRIS] * len(med)
    colores[-1] = AZUL
    fig, ax = plt.subplots(figsize=(6, 3.6))
    y = np.arange(len(med))
    ax.barh(y, med.to_numpy(), color=colores, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(med.index)
    ax.set_xlabel("popularidad media")
    ax.set_xlim(0, med.max() * 1.15)
    ax.axvline(POP_MEDIA, color=TINTA, ls=":", lw=1)
    ax.text(POP_MEDIA, len(med) - 0.4, "media global",
            color=TINTA, fontsize=8, ha="center")
    for yi, v in zip(y, med.to_numpy(), strict=False):
        ax.text(v + 0.4, yi, f"{v:.1f}", va="center",
                color=TINTA, fontsize=9)
    ax.set_title("El pop lidera la popularidad media")
    return guardar(fig, salida)


def fig06_agrupadas(salida):
    """Dos factores: media de cada rasgo agrupada por genero."""
    d = DF[DF["track_genre"].isin(GEN3)]
    tabla = d.pivot_table(index="track_genre", columns=None,
                          values=FEAT3, aggfunc="mean")
    tabla = tabla.reindex(index=GEN3)[FEAT3].round(3)
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    x = np.arange(len(GEN3))
    w = 0.26
    for k, feat in enumerate(FEAT3):
        ax.bar(x + (k - 1) * w, tabla[feat].to_numpy(), width=w,
               color=COLOR_FEAT[feat], label=feat)
    ax.set_xticks(x)
    ax.set_xticklabels(GEN3)
    ax.set_ylabel("valor medio del rasgo (0–1)")
    ax.set_xlabel("género")
    ax.legend(title="rasgo")
    ax.set_title("Media de cada rasgo por género")
    return guardar(fig, salida)


def fig07_histograma(salida):
    """Distribucion de danceability: histograma + KDE, media y mediana."""
    dance = DF["danceability"].to_numpy()
    media = float(np.mean(dance))
    mediana = float(np.median(dance))
    kde = stats.gaussian_kde(dance)
    xs = np.linspace(dance.min(), dance.max(), 200)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(dance, bins=28, density=True, color="#bcd6ea",
            edgecolor="white", linewidth=0.5)
    ax.plot(xs, kde(xs), color=AZUL, lw=2)
    ax.axvline(media, color=BERMELLON, ls="--", lw=1.5)
    ax.axvline(mediana, color=VERDE, ls="--", lw=1.5)
    ytop = ax.get_ylim()[1]
    ax.annotate(f"media {media:.3f}", xy=(media, ytop * 0.9),
                xytext=(media - 0.42, ytop * 0.92), color=TINTA,
                fontsize=9,
                arrowprops={"arrowstyle": "->",
                            "color": BERMELLON, "lw": 1})
    ax.annotate(f"mediana {mediana:.3f}",
                xy=(mediana, ytop * 0.72),
                xytext=(mediana + 0.12, ytop * 0.74), color=TINTA,
                fontsize=9,
                arrowprops={"arrowstyle": "->",
                            "color": VERDE, "lw": 1})
    ax.set_xlabel("danceability")
    ax.set_ylabel("densidad")
    ax.set_title("Distribución de danceability")
    return guardar(fig, salida)


def fig08_caja_violin(salida):
    """Caja frente a violin: el violin anade la forma."""
    datos = [DF[f].to_numpy() for f in FEAT3]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    bp = a.boxplot(datos, patch_artist=True, widths=0.55,
                   medianprops={"color": TINTA})
    for parche, f in zip(bp["boxes"], FEAT3, strict=False):
        parche.set_facecolor(COLOR_FEAT[f])
        parche.set_alpha(0.85)
        parche.set_edgecolor(TINTA)
    a.set_xticks([1, 2, 3])
    a.set_xticklabels(FEAT3)
    a.set_ylabel("valor del rasgo (0–1)")
    a.set_title("Caja: resumen de cinco números")
    vp = b.violinplot(datos, showmedians=True)
    for cuerpo, f in zip(vp["bodies"], FEAT3, strict=False):
        cuerpo.set_facecolor(COLOR_FEAT[f])
        cuerpo.set_alpha(0.75)
        cuerpo.set_edgecolor(TINTA)
    for clave in ("cbars", "cmins", "cmaxes", "cmedians"):
        vp[clave].set_color(TINTA)
        vp[clave].set_linewidth(1)
    b.set_xticks([1, 2, 3])
    b.set_xticklabels(FEAT3)
    b.set_title("Violín: además, la forma")
    return guardar(fig, salida)


def fig09_ecdf(salida):
    """ECDF de los tres rasgos; la horizontal marca la mediana."""
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for f in FEAT3:
        x, y = ecdf(DF[f].to_numpy())
        ax.step(x, y, where="post", color=COLOR_FEAT[f], lw=2,
                label=f)
    ax.axhline(0.5, color=GRIS, ls=":", lw=1)
    ax.text(ax.get_xlim()[1], 0.5, " mediana", va="center",
            color=TINTA, fontsize=8)
    ax.set_xlabel("valor del rasgo (0–1)")
    ax.set_ylabel("proporción acumulada")
    ax.set_ylim(0, 1.02)
    ax.legend(title="rasgo")
    ax.set_title("ECDF: leer percentiles de un vistazo")
    return guardar(fig, salida)


def fig10_anscombe(salida):
    """Cuarteto de Anscombe: mismas estadisticas, formas distintas."""
    x = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], dtype=float)
    x4 = np.array([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8], dtype=float)
    ys = {
        "I": [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26,
              10.84, 4.82, 5.68],
        "II": [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10,
               9.13, 7.26, 4.74],
        "III": [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39,
                8.15, 6.42, 5.73],
        "IV": [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50,
               5.56, 7.91, 6.89],
    }
    xs = {"I": x, "II": x, "III": x, "IV": x4}
    fig, axs = plt.subplots(2, 2, figsize=(7.5, 6),
                            sharex=True, sharey=True)
    recta = np.array([2.5, 20.0])
    for ax, k in zip(axs.flat, ys, strict=False):
        xv, yv = xs[k], np.array(ys[k])
        m, b0 = np.polyfit(xv, yv, 1)
        r = np.corrcoef(xv, yv)[0, 1]
        ax.plot(recta, m * recta + b0, color=BERMELLON, lw=1.5)
        ax.plot(xv, yv, "o", color=AZUL, ms=7)
        ax.set_title(f"Conjunto {k}")
        ax.annotate(f"r = {r:.2f}\ny = {b0:.1f} + {m:.2f}·x",
                    xy=(0.05, 0.9), xycoords="axes fraction",
                    fontsize=8, color=TINTA, va="top")
    fig.supxlabel("x")
    fig.supylabel("y")
    fig.suptitle("El cuarteto de Anscombe: mismas estadísticas",
                 fontsize=12, fontweight="bold")
    return guardar(fig, salida, rect=(0, 0, 1, 0.95))


def fig11_dispersion(salida):
    """Dispersion popularidad-energy con hexbin; anota la correlacion."""
    x = DF["energy"].to_numpy()
    y = DF["popularity"].to_numpy()
    r = np.corrcoef(x, y)[0, 1]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    hb = ax.hexbin(x, y, gridsize=28, cmap="viridis", mincnt=1)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("nº de pistas")
    ax.set_xlabel("energy (0–1)")
    ax.set_ylabel("popularidad (0–100)")
    ax.annotate(f"r = {r:.2f} (sin relación aparente)",
                xy=(0.04, 0.96), xycoords="axes fraction",
                color="black", fontsize=9, va="top",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    ax.set_title("Popularidad frente a energy, pista a pista")
    ax.grid(False)
    return guardar(fig, salida)


def fig12_heatmap(salida):
    """Mapa de calor de las correlaciones entre rasgos de audio."""
    feats = ["danceability", "energy", "loudness", "speechiness",
             "acousticness", "instrumentalness", "liveness",
             "valence", "tempo"]
    corr = DF[feats].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    # mapa divergente (RdBu, apto para daltonicos): el cero tiene sentido
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("correlación (r)")
    ax.set_xticks(range(len(feats)))
    ax.set_yticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(feats, fontsize=8)
    for i in range(len(feats)):
        for j in range(len(feats)):
            v = corr[i, j]
            col = "white" if abs(v) > 0.55 else TINTA
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=col, fontsize=7)
    ax.set_title("Correlaciones entre rasgos de audio")
    ax.grid(False)
    return guardar(fig, salida)


def fig13_serie(salida):
    """Perfil ordenado: loudness medio por nivel de popularidad."""
    perfil = perfil_loudness()
    suave = perfil.rolling(7, center=True, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.plot(perfil.index, perfil.to_numpy(), color=GRIS, lw=0.9,
            label="media por nivel")
    ax.plot(suave.index, suave.to_numpy(), color=AZUL, lw=2.2,
            label="suavizado local (7)")
    ax.axhline(DF["loudness"].mean(), color=TINTA, ls=":", lw=1)
    ax.set_xlabel("popularidad (0–100)")
    ax.set_ylabel("loudness medio (dB)")
    ax.legend()
    ax.set_title("Loudness medio por nivel de popularidad")
    return guardar(fig, salida)


def fig14_seaborn(salida):
    """La gramatica de graficos: el mismo violin con seaborn."""
    largo = DF[FEAT3].melt(var_name="rasgo", value_name="valor")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    sns.violinplot(data=largo, x="rasgo", y="valor",
                   order=FEAT3, hue="rasgo",
                   palette=COLOR_FEAT, legend=False, inner="box",
                   ax=ax)
    ax.set_xlabel("rasgo")
    ax.set_ylabel("valor (0–1)")
    ax.set_title("Lo mismo con seaborn, en una línea")
    return guardar(fig, salida)


def fig15_comunicacion(salida):
    """Antes/despues: exploracion cruda frente a comunicacion."""
    perfil = perfil_loudness()
    cima_x = int(perfil.idxmax())
    cima_y = float(perfil.max())
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.plot(perfil.index, perfil.to_numpy(), color=GRIS, lw=1)
    a.set_xlabel("popularidad")
    a.set_title("Antes: exploración cruda", color=GRIS,
                fontweight="normal", fontsize=10)
    b.plot(perfil.index, perfil.to_numpy(), color=AZUL, lw=1.6)
    b.axhline(DF["loudness"].mean(), color=TINTA, ls=":", lw=1)
    etiqueta = (f"máximo local\n{cima_y:.1f} dB\n"
                f"popularidad {cima_x}")
    b.annotate(etiqueta, xy=(cima_x, cima_y),
               xytext=(cima_x - 55, cima_y - 2.2),
               color=TINTA, fontsize=8.5,
               arrowprops={"arrowstyle": "->", "color": TINTA,
                           "lw": 1})
    b.set_xlabel("popularidad")
    b.set_ylabel("loudness medio (dB)")
    b.set_title("El volumen sube algo con la popularidad",
                loc="left")
    n = int(DF.shape[0])
    fig.text(0.99, 0.01,
             f"n = {n} pistas · fuente: Spotify Tracks (maharshipandya)",
             ha="right", va="bottom", color=GRIS, fontsize=8)
    return guardar(fig, salida, rect=(0, 0.05, 1, 1))


def fig16_panel(salida):
    """Integrador: el catalogo musical en cuatro vistas (2x2)."""
    fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=(10, 7.5))
    # (a) perfil ordenado: loudness por nivel de popularidad
    perfil = perfil_loudness()
    suave = perfil.rolling(7, center=True, min_periods=1).mean()
    a.plot(perfil.index, perfil.to_numpy(), color=GRIS, lw=0.8)
    a.plot(suave.index, suave.to_numpy(), color=AZUL, lw=2)
    a.set_xlabel("popularidad")
    a.set_ylabel("loudness (dB)")
    a.set_title("(a) Loudness por popularidad")
    # (b) distribucion por rasgo (caja)
    datos = [DF[f].to_numpy() for f in FEAT3]
    bp = b.boxplot(datos, patch_artist=True, widths=0.55,
                   medianprops={"color": TINTA})
    for parche, f in zip(bp["boxes"], FEAT3, strict=False):
        parche.set_facecolor(COLOR_FEAT[f])
        parche.set_alpha(0.85)
        parche.set_edgecolor(TINTA)
    b.set_xticks([1, 2, 3])
    b.set_xticklabels(FEAT3)
    b.set_ylabel("valor (0–1)")
    b.set_title("(b) Distribución por rasgo")
    # (c) barras popularidad por genero
    med = pop_por_genero(PUNADO).round(1)
    colores = [GRIS] * len(med)
    colores[-1] = AZUL
    yb = np.arange(len(med))
    c.barh(yb, med.to_numpy(), color=colores, height=0.6)
    c.set_yticks(yb)
    c.set_yticklabels(med.index, fontsize=8)
    c.set_xlabel("popularidad media")
    c.set_xlim(0, med.max() * 1.12)
    c.set_title("(c) Popularidad media por género")
    # (d) ECDF por rasgo
    for f in FEAT3:
        xe, ye = ecdf(DF[f].to_numpy())
        d.step(xe, ye, where="post", color=COLOR_FEAT[f], lw=1.8,
               label=f)
    d.axhline(0.5, color=GRIS, ls=":", lw=1)
    d.set_xlabel("valor del rasgo (0–1)")
    d.set_ylabel("prop. acumulada")
    d.legend(title="rasgo", fontsize=8)
    d.set_title("(d) ECDF por rasgo")
    fig.suptitle("La música en cuatro vistas",
                 fontsize=13, fontweight="bold")
    return guardar(fig, salida, rect=(0, 0, 1, 0.96))


FIGURAS = [
    ("cap12_jerarquia", fig01_jerarquia),
    ("cap12_tarta_barras", fig02_tarta_barras),
    ("cap12_liefactor", fig03_liefactor),
    ("cap12_anatomia", fig04_anatomia),
    ("cap12_barras", fig05_barras),
    ("cap12_agrupadas", fig06_agrupadas),
    ("cap12_histograma", fig07_histograma),
    ("cap12_caja_violin", fig08_caja_violin),
    ("cap12_ecdf", fig09_ecdf),
    ("cap12_anscombe", fig10_anscombe),
    ("cap12_dispersion", fig11_dispersion),
    ("cap12_heatmap", fig12_heatmap),
    ("cap12_serie", fig13_serie),
    ("cap12_seaborn", fig14_seaborn),
    ("cap12_comunicacion", fig15_comunicacion),
    ("cap12_panel", fig16_panel),
]


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for nombre, func in FIGURAS:
        size = func(nombre)
        print(f"generada: {nombre} ({size} bytes)")


if __name__ == "__main__":
    main()
