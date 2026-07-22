"""Código reproducible del cap. 15 «Privacidad y confidencialidad».

Genera un conjunto de PERFILES DE ESCUCHA SINTÉTICOS (declarado Clase 2, no hay
personas reales) de un servicio de streaming ficticio: cada usuario vive en un
país, tiene un código postal, un gusto musical (género más escuchado y perfil
acústico), una huella de escucha ---el hash de sus artistas top, casi único como
la huella de un navegador--- y una etiqueta comercial sensible (premium) que
depende de la intensidad de escucha, el perfil y la edad. Los rasgos musicales se
muestrean de las distribuciones REALES por género de ``data/processed/musica.parquet``.
Sobre él se ilustran, ejecutando de verdad: la reidentificación por
cuasi-identificadores, el k-anonimato, la entropía en bits (la huella de escucha),
el mecanismo de Laplace de la privacidad diferencial, la inferencia de pertenencia
contra un modelo y la utilidad de los datos sintéticos.

Uso:
    python src/cap15_privacidad.py --cifras     # imprime las cifras del texto
    python src/cap15_privacidad.py --figuras    # regenera las figuras
    python src/cap15_privacidad.py              # ambas

Entorno verificado: scikit-learn 1.7.2, numpy 2.3.3, pandas 2.3.3 (CPython 3.11).
Ni diffprivlib (IBM) ni Opacus (Meta) se usan aquí: se citan como ecosistema.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

RAIZ = Path(__file__).resolve().parents[1]
PARQUET = RAIZ / "data" / "processed" / "perfiles_escucha.parquet"
MUSICA = RAIZ / "data" / "processed" / "musica.parquet"
FIG = RAIZ / "latex" / "figures"
SEMILLA = 2026

# Veinte países del servicio, con su peso de población de usuarios (más grandes
# los mercados grandes). Los gustos difieren por país: cada uno realza un puñado
# de géneros firma, tomados del catálogo real de musica.parquet. [sintético]
PAISES = {
    "España": 0.06, "México": 0.09, "Argentina": 0.05, "Colombia": 0.05,
    "Chile": 0.03, "Estados Unidos": 0.10, "Brasil": 0.08, "Reino Unido": 0.06,
    "Francia": 0.05, "Alemania": 0.06, "Italia": 0.04, "Japón": 0.05,
    "Corea del Sur": 0.03, "India": 0.05, "Nigeria": 0.03, "Suecia": 0.02,
    "Canadá": 0.03, "Australia": 0.02, "Países Bajos": 0.03, "Portugal": 0.02,
}

# géneros firma por país (existen con esta grafía exacta en musica.parquet)
GUSTOS = {
    "España": ["spanish", "reggaeton", "pop", "salsa"],
    "México": ["latino", "reggaeton", "salsa", "latin"],
    "Argentina": ["tango", "latin", "rock", "reggaeton"],
    "Colombia": ["reggaeton", "salsa", "latino", "dancehall"],
    "Chile": ["latin", "reggaeton", "indie", "pop"],
    "Estados Unidos": ["hip-hop", "pop", "country", "r-n-b"],
    "Brasil": ["brazil", "samba", "mpb", "sertanejo"],
    "Reino Unido": ["british", "rock", "indie", "punk"],
    "Francia": ["french", "house", "electronic", "disco"],
    "Alemania": ["german", "techno", "minimal-techno", "trance"],
    "Italia": ["opera", "pop", "classical", "disco"],
    "Japón": ["j-pop", "j-rock", "anime", "j-idol"],
    "Corea del Sur": ["k-pop", "pop", "r-n-b", "dance"],
    "India": ["indian", "world-music", "pop", "romance"],
    "Nigeria": ["afrobeat", "dancehall", "reggae", "world-music"],
    "Suecia": ["swedish", "synth-pop", "edm", "house"],
    "Canadá": ["pop", "rock", "hip-hop", "country"],
    "Australia": ["rock", "alt-rock", "indie", "pop"],
    "Países Bajos": ["trance", "hardstyle", "house", "techno"],
    "Portugal": ["pop", "world-music", "rock", "romance"],
}


def _perfil_musical(seed: int = 0):
    """Estadísticos reales por género (media y desviación de energy y valence) y
    el vivero de artistas de cada género, leídos de musica.parquet. Alimentan el
    muestreo verosímil de los perfiles de escucha (regla de coherencia)."""
    m = pd.read_parquet(MUSICA, columns=["artists", "energy", "valence",
                                         "track_genre"])
    stats = m.groupby("track_genre").agg(
        e_m=("energy", "mean"), e_s=("energy", "std"),
        v_m=("valence", "mean"), v_s=("valence", "std"))
    primero = m["artists"].astype(str).str.split(";").str[0].str.strip()
    vivero = {g: sub.to_numpy()
              for g, sub in primero.groupby(m["track_genre"].to_numpy())}
    return stats, vivero


def _huella_escucha(genero, n_art, vivero, rng) -> list[str]:
    """Huella de escucha: hash de los ~50 artistas top de cada usuario, tomados
    de su género. El conjunto es casi único ---como la huella de un navegador---,
    de modo que individúa sin nombre."""
    out = []
    for g, k in zip(genero, n_art):
        pool = vivero[g]
        top = rng.choice(pool, size=int(min(max(k, 5), 50)), replace=True)
        firma = "|".join(sorted(set(top.tolist())))
        out.append(hashlib.sha256(firma.encode()).hexdigest()[:16])
    return out


def generar(n: int = 50000, seed: int = 42) -> pd.DataFrame:
    """Perfiles de escucha sintéticos (Clase 2). Semilla 42 como el resto del
    libro. Los rasgos musicales salen de las distribuciones reales por género."""
    rng = np.random.default_rng(seed)
    stats, vivero = _perfil_musical()
    generos = list(stats.index)
    idx_gen = {g: i for i, g in enumerate(generos)}
    paises = list(PAISES)
    pesos_pais = np.array(list(PAISES.values()))
    pesos_pais = pesos_pais / pesos_pais.sum()
    pais = rng.choice(paises, size=n, p=pesos_pais)
    # código postal: 6 por país (no solapan), cuasi-identificador fino regional
    base_cp = {p: 10000 + 500 * i for i, p in enumerate(paises)}
    cp = np.array([base_cp[p] + rng.integers(0, 6) for p in pais])
    # edad 0-99 con forma de pirámide suave; sexo
    edad = np.clip(rng.gamma(shape=7.0, scale=6.0, size=n), 0, 99).astype(int)
    # fecha de nacimiento: la edad en años más un día del año -> identificador
    # fino (como en el caso de Sweeney). Fecha de referencia 1-ene-2026.
    dias = edad * 365 + rng.integers(0, 365, n)
    fecha_nac = pd.Timestamp("2026-01-01") - pd.to_timedelta(dias, unit="D")
    sexo = rng.choice(["M", "F"], size=n)

    # género más escuchado: base leve global + realce de los géneros firma del
    # país, de modo que los gustos difieran por país [señal sintética]
    base = np.ones(len(generos))
    genero_top = np.empty(n, dtype=object)
    for p in paises:
        w = base.copy()
        for g in GUSTOS[p]:
            w[idx_gen[g]] += 12.0
        w = w / w.sum()
        m = pais == p
        genero_top[m] = rng.choice(generos, size=int(m.sum()), p=w)

    # perfil acústico del gusto: media del género real + ruido (regla de
    # coherencia). Es el análogo continuo del rasgo ligado a la región.
    e_m = stats["e_m"].to_dict(); e_s = stats["e_s"].to_dict()
    v_m = stats["v_m"].to_dict(); v_s = stats["v_s"].to_dict()
    energia = np.array([rng.normal(e_m[g], 0.5 * e_s[g]) for g in genero_top])
    valence = np.array([rng.normal(v_m[g], 0.5 * v_s[g]) for g in genero_top])
    energia = np.clip(energia, 0, 1)
    valence = np.clip(valence, 0, 1)

    # intensidad y amplitud de escucha, y grado del grafo social (seguidos)
    minutos = np.clip(rng.gamma(shape=2.0, scale=45.0, size=n), 0, 600)
    n_art = np.clip(rng.gamma(shape=4.0, scale=10.0, size=n), 3, 300).astype(int)
    sigue_a = rng.zipf(2.2, n).clip(max=5000)

    # huella de escucha: hash de los artistas top de cada usuario
    huella = _huella_escucha(genero_top, n_art, vivero, rng)

    # premium: logística sobre la intensidad de escucha, el perfil acústico y la
    # edad [señal sintética]. Los oyentes intensos y de cierto perfil pagan más.
    z = (-2.75 + 0.95 * (minutos - 90) / 60 + 0.55 * (energia - 0.5)
         + 0.45 * (edad - 35) / 20 + 0.30 * (n_art - 40) / 40
         + 0.15 * (sexo == "F"))
    premium = rng.random(n) < 1 / (1 + np.exp(-z))
    return pd.DataFrame({
        "id": np.arange(1, n + 1),
        "fecha_nac": fecha_nac, "edad": edad, "sexo": sexo,
        "pais": pais, "cp": cp,
        "genero_top": genero_top,
        "energia_media": np.round(energia, 3),
        "valence_media": np.round(valence, 3),
        "minutos_dia": minutos.astype(int),
        "n_artistas": n_art, "sigue_a": sigue_a.astype(int),
        "huella_top50": huella,
        "premium": premium.astype(int),
    })


def carga() -> pd.DataFrame:
    if PARQUET.exists():
        return pd.read_parquet(PARQUET)
    df = generar()
    PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET)
    return df


# --- 15.2 Reidentificación y k-anonimato -------------------------------------

def _k_por_grupo(df: pd.DataFrame, cuasi: list[str]) -> pd.Series:
    """Tamaño del grupo (k) al que pertenece cada registro según los cuasi-id."""
    return df.groupby(cuasi)[cuasi[0]].transform("size")


def reidentificacion(df: pd.DataFrame) -> None:
    for cuasi in (["cp", "fecha_nac", "sexo"], ["cp", "edad", "sexo"],
                  ["pais", "edad", "sexo"]):
        k = _k_por_grupo(df, cuasi)
        unicos = (k == 1).mean()
        print(f"       {'+'.join(cuasi):>22}: {unicos*100:5.1f}% únicos, "
              f"k mínimo={k.min()}")


def _generaliza(df: pd.DataFrame) -> pd.DataFrame:
    """Generaliza edad a decenios y sustituye el CP y la fecha por el país."""
    g = df.copy()
    g["edad"] = (g["edad"] // 10 * 10)         # 0-9, 10-19, ...
    g = g.drop(columns=["cp", "fecha_nac"])
    return g


def entropia_huella(df: pd.DataFrame) -> None:
    """Conteo de bits (entropia) y unicidad para cuasi-identificadores de
    detalle creciente: demografico -> + grafo social (sigue_a) -> + huella de
    escucha (huella_top50). La huella satura el techo log2(N)."""
    d = df.copy()
    n = len(d)
    d["edad"] = d["edad"] // 10 * 10                 # generalizada a decenios

    def bits_unicos(quasi):
        tam = d.groupby(quasi)[d.columns[0]].transform("size")
        p = d.groupby(quasi).size() / n
        return float(-(p * np.log2(p)).sum()), 100 * float((tam == 1).mean())

    escalera = {
        "demografico {pais,edad,sexo}": ["pais", "edad", "sexo"],
        "+ grado del grafo social": ["pais", "edad", "sexo", "sigue_a"],
        "+ huella de escucha (top-50)": ["pais", "edad", "sexo", "huella_top50"],
    }
    print(f"       techo teorico log2(N) = {np.log2(n):.1f} bits")
    for nombre, quasi in escalera.items():
        h, u = bits_unicos(quasi)
        print(f"         {nombre:>34}: {h:5.1f} bits | {u:5.1f}% unicos")


def k_anonimato(df: pd.DataFrame) -> None:
    cuasi_fino = ["cp", "fecha_nac", "sexo"]
    k0 = _k_por_grupo(df, cuasi_fino)
    print(f"       antes (cp+fecha+sexo): {(k0 < 5).mean()*100:.1f}% en grupos k<5")
    g = _generaliza(df)
    k1 = _k_por_grupo(g, ["pais", "edad", "sexo"])
    print(f"       tras generalizar     : {(k1 < 5).mean()*100:.1f}% en grupos "
          f"k<5, k mínimo={k1.min()}")
    # l-diversidad: diversidad de 'premium' dentro de cada grupo k-anónimo
    div = g.groupby(["pais", "edad", "sexo"])["premium"].nunique()
    print(f"       l-diversidad: {(div >= 2).mean()*100:.1f}% de los grupos "
          f"tienen ambas clases de premium")


# --- 15.3 Privacidad diferencial: mecanismo de Laplace ------------------------

def laplace_conteo(df: pd.DataFrame) -> None:
    """Consulta '¿cuántos usuarios premium por país?' con ruido de Laplace."""
    rng = np.random.default_rng(SEMILLA)
    real = df.groupby("pais")["premium"].sum()
    sens = 1.0  # un usuario cambia el conteo como mucho en 1
    print("       consulta: nº de usuarios premium por país (sensibilidad 1)")
    for eps in (0.1, 0.5, 1.0, 5.0):
        # promediamos el error absoluto sobre 200 repeticiones del mecanismo
        errs = []
        for _ in range(200):
            ruido = rng.laplace(0, sens / eps, size=len(real))
            errs.append(np.abs(ruido).mean())
        print(f"         epsilon={eps:>4}: error absoluto medio "
              f"~ {np.mean(errs):6.2f} usuarios")


def ecosistema_diffprivlib(df: pd.DataFrame) -> None:
    """Un conteo con privacidad diferencial usando diffprivlib (IBM), la misma
    idea que el Laplace a mano pero con la contabilidad del presupuesto ya hecha."""
    try:
        from diffprivlib.mechanisms import Laplace
    except ModuleNotFoundError:
        print("       (diffprivlib no disponible; se omite el ejemplo)")
        return
    real = int(df["premium"].sum())
    dp = Laplace(epsilon=1.0, sensitivity=1, random_state=SEMILLA).randomise(real)
    print("       diffprivlib (IBM), conteo con privacidad diferencial:")
    print(f"         total real de usuarios premium = {real}")
    print(f"         conteo con DP (epsilon=1) = {dp:.2f}")


def respuesta_aleatorizada(df: pd.DataFrame) -> None:
    """Privacidad diferencial LOCAL (Warner): cada persona perturba su propia
    respuesta con una moneda, pero el agregado se recupera insesgado."""
    rng = np.random.default_rng(SEMILLA)
    prem = df["premium"].to_numpy().astype(bool)
    cara1 = rng.random(len(prem)) < 0.5   # cara: responde la verdad
    cara2 = rng.random(len(prem)) < 0.5   # cruz: responde al azar
    resp = np.where(cara1, prem, cara2)
    obs = resp.mean()
    p_hat = 2 * obs - 0.5                  # estimador insesgado
    print("       respuesta aleatorizada (privacidad diferencial local):")
    print(f"         tasa real de premium    = {prem.mean():.4f}")
    print(f"         tasa observada          = {obs:.4f}")
    print(f"         estimacion corregida    = {p_hat:.4f} "
          f"(error {abs(p_hat - prem.mean()):.4f})")
    print(f"         garantia epsilon = ln 3 = {np.log(3):.4f}")


# --- 15.4 Inferencia de pertenencia -------------------------------------------

def _datos_mia(df: pd.DataFrame):
    X = df[["edad", "cp", "energia_media", "minutos_dia"]].copy()
    X["sexo"] = (df["sexo"] == "F").astype(int)
    return X, df["premium"].to_numpy()


# configuraciones de modelo: uno de alta capacidad (memoriza) y uno regularizado
MIA_OVERFIT = dict(max_depth=None, max_iter=600, min_samples_leaf=1,
                   l2_regularization=0.0, learning_rate=0.3)
MIA_REGULAR = dict(max_depth=3, max_iter=100, min_samples_leaf=100,
                   l2_regularization=1.0)


def inferencia_pertenencia(df: pd.DataFrame) -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    X, y = _datos_mia(df)
    # semilla 42: la misma con que el capítulo muestra el ataque de pertenencia.
    # entrenamos con POCOS datos y alta capacidad: así el modelo memoriza
    Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=2000, test_size=5000,
                                          random_state=42, stratify=y)
    for etiqueta, kw in [("sobreajustado", MIA_OVERFIT),
                         ("regularizado", MIA_REGULAR)]:
        clf = HistGradientBoostingClassifier(random_state=42, **kw).fit(Xtr, ytr)
        gap = clf.score(Xtr, ytr) - clf.score(Xte, yte)
        # ataque: la confianza del modelo en la clase real distingue a los
        # miembros del entrenamiento (probabilidad más alta) de los ajenos
        p_tr = clf.predict_proba(Xtr)[np.arange(len(ytr)), ytr]
        p_te = clf.predict_proba(Xte)[np.arange(len(yte)), yte]
        etiquetas = np.r_[np.ones_like(p_tr), np.zeros_like(p_te)]
        auc = roc_auc_score(etiquetas, np.r_[p_tr, p_te])
        print(f"       {etiqueta:>14}: brecha train-test="
              f"{gap:+.3f}  AUC del ataque={auc:.3f}")


# --- 15.5 Datos sintéticos: utilidad ------------------------------------------

def _sintetiza(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Generador ingenuo: muestrea cada columna de su marginal (rompe la
    correlación). Sirve para ilustrar el compromiso utilidad/privacidad."""
    rng = np.random.default_rng(seed)
    n = len(df)
    out = {c: df[c].sample(n, replace=True, random_state=seed).to_numpy()
           for c in ["edad", "sexo", "cp", "energia_media", "minutos_dia"]}
    # la etiqueta se re-muestrea aparte: el sintético ingenuo pierde la relación
    out["premium"] = df["premium"].sample(n, replace=True,
                                          random_state=seed + 1).to_numpy()
    return pd.DataFrame(out)


def datos_sinteticos(df: pd.DataFrame) -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    def prep(d):
        X = d[["edad", "sexo", "cp", "energia_media", "minutos_dia"]].copy()
        X["sexo"] = (X["sexo"] == "F").astype(int)
        return X, d["premium"].to_numpy()

    Xtr, Xte, ytr, yte = train_test_split(*prep(df), test_size=0.3,
                                          random_state=SEMILLA)
    real = HistGradientBoostingClassifier(random_state=SEMILLA).fit(Xtr, ytr)
    auc_real = roc_auc_score(yte, real.predict_proba(Xte)[:, 1])
    sint = _sintetiza(df, SEMILLA)
    Xs, ys = prep(sint)
    modelo_s = HistGradientBoostingClassifier(random_state=SEMILLA).fit(Xs, ys)
    auc_sint = roc_auc_score(yte, modelo_s.predict_proba(Xte)[:, 1])
    print(f"       entrenar con datos REALES     -> AUC en test real {auc_real:.3f}")
    print(f"       entrenar con SINTÉTICO ingenuo -> AUC en test real {auc_sint:.3f}")
    print("       (el sintético que rompe la correlación pierde utilidad: por eso")
    print("        un buen generador debe preservarla, y con DP para no filtrar)")


def cifras() -> None:
    df = carga()
    print("perfiles de escucha:", df.shape, "| tasa de premium =",
          f"{df['premium'].mean():.3f}")
    print("[15.2] reidentificación por cuasi-identificadores:")
    reidentificacion(df)
    print("[15.2] k-anonimato y l-diversidad:")
    k_anonimato(df)
    print("[15.2] entropia: cuasi-identificador demografico vs huella de escucha:")
    entropia_huella(df)
    print("[15.3] privacidad diferencial (mecanismo de Laplace):")
    laplace_conteo(df)
    respuesta_aleatorizada(df)
    ecosistema_diffprivlib(df)
    print("[15.4] inferencia de pertenencia (el sobreajuste filtra):")
    inferencia_pertenencia(df)
    print("[15.5] datos sintéticos (utilidad):")
    datos_sinteticos(df)


def figuras() -> None:
    import matplotlib as mpl
    mpl.use("Agg")
    sys.path.insert(0, str(RAIZ / "src"))
    from cap12_estilo import AZUL, BERMELLON, GRIS, NARANJA, TINTA, aplicar  # noqa: E402
    aplicar()
    import matplotlib.pyplot as plt
    df = carga()
    FIG.mkdir(parents=True, exist_ok=True)

    # cap15_reident: % de registros únicos (reidentificables) según lo fino que
    # sea el cuasi-identificador
    cuasis = [("cp + fecha nac.\n+ sexo", ["cp", "fecha_nac", "sexo"]),
              ("cp + edad\n+ sexo", ["cp", "edad", "sexo"]),
              ("país + edad\n+ sexo", ["pais", "edad", "sexo"])]
    pcts = [(_k_por_grupo(df, c) == 1).mean() * 100 for _, c in cuasis]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.bar([n for n, _ in cuasis], pcts,
           color=[BERMELLON, NARANJA, AZUL], width=0.62)
    for i, p in enumerate(pcts):
        ax.text(i, p + 2, f"{p:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("registros únicos, reidentificables (%)")
    ax.set_ylim(0, 112)
    ax.set_title("Cuanto más fino el cuasi-identificador, más se reidentifica")
    fig.tight_layout()
    fig.savefig(FIG / "cap15_reident.pdf", bbox_inches="tight"); plt.close(fig)

    # cap15_epsilon: error del mecanismo de Laplace frente a epsilon
    rng = np.random.default_rng(SEMILLA)
    real = df.groupby("pais")["premium"].sum()
    eps = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
    err = [np.mean([np.abs(rng.laplace(0, 1 / e, len(real))).mean()
                    for _ in range(200)]) for e in eps]
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(eps, err, "o-", color=AZUL)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("presupuesto de privacidad $\\varepsilon$ (menor = más privado)")
    ax.set_ylabel("error medio del conteo (usuarios)")
    ax.set_title("El precio de la privacidad diferencial")
    fig.tight_layout(); fig.savefig(FIG / "cap15_epsilon.pdf"); plt.close(fig)

    # cap15_mia: ROC del ATAQUE de pertenencia para dos modelos (el que
    # memoriza filtra; el regularizado apenas se distingue del azar)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import train_test_split
    X, y = _datos_mia(df)
    # semilla 42 aquí, para casar con el listado de inferencia del capítulo
    Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=2000, test_size=5000,
                                          random_state=42, stratify=y)
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    for tag, kw, col in [("modelo sobreajustado", MIA_OVERFIT, BERMELLON),
                         ("modelo regularizado", MIA_REGULAR, AZUL)]:
        clf = HistGradientBoostingClassifier(random_state=42, **kw).fit(Xtr, ytr)
        p_tr = clf.predict_proba(Xtr)[np.arange(len(ytr)), ytr]
        p_te = clf.predict_proba(Xte)[np.arange(len(yte)), yte]
        etiq = np.r_[np.ones_like(p_tr), np.zeros_like(p_te)]
        sc = np.r_[p_tr, p_te]
        fpr, tpr, _ = roc_curve(etiq, sc)
        ax.plot(fpr, tpr, color=col,
                label=f"{tag} (AUC={roc_auc_score(etiq, sc):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1)
    ax.set_xlabel("tasa de falsos positivos")
    ax.set_ylabel("tasa de aciertos del ataque")
    ax.set_title("Ataque de pertenencia: el que memoriza filtra")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "cap15_mia.pdf"); plt.close(fig)

    # cap15_dp_hist: la privacidad diferencial en acción. Distribución de la
    # respuesta con ruido para UNA consulta (premium de un país), a dos
    # presupuestos: epsilon=1 se concentra en el valor real; epsilon=0.1 se
    # dispersa (más privacidad, menos exactitud).
    rng2 = np.random.default_rng(SEMILLA)
    verdad = int(df.groupby("pais")["premium"].sum().median())
    m = 20000
    r1 = verdad + rng2.laplace(0, 1 / 1.0, m)
    r01 = verdad + rng2.laplace(0, 1 / 0.1, m)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    binss = np.linspace(verdad - 45, verdad + 45, 90)
    ax.hist(r01, bins=binss, density=True, color=BERMELLON, alpha=0.75,
            label="$\\varepsilon=0{,}1$ (más privado)")
    ax.hist(r1, bins=binss, density=True, color=AZUL, alpha=0.8,
            label="$\\varepsilon=1$")
    ax.axvline(verdad, color="0.3", lw=1.5, ls="--",
               label=f"valor real ({verdad})")
    ax.set_xlabel("respuesta publicada (nº de usuarios premium)")
    ax.set_ylabel("densidad")
    ax.set_title("Privacidad diferencial: la misma consulta, dos presupuestos")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "cap15_dp_hist.pdf"); plt.close(fig)

    # cap15_sintetico: utilidad de un modelo entrenado con datos reales frente
    # a uno entrenado con un sintetico ingenuo (que rompe las correlaciones).
    from sklearn.ensemble import HistGradientBoostingClassifier as _HGBC
    from sklearn.metrics import roc_auc_score as _auc
    from sklearn.model_selection import train_test_split as _tts

    def _prep(d):
        Xd = d[["edad", "cp", "energia_media", "minutos_dia"]].copy()
        Xd["sexo"] = (d["sexo"] == "F").astype(int)
        return Xd, d["premium"].to_numpy()
    Xtr2, Xte2, ytr2, yte2 = _tts(*_prep(df), test_size=0.3, random_state=SEMILLA)
    auc_real = _auc(yte2, _HGBC(random_state=SEMILLA).fit(Xtr2, ytr2)
                    .predict_proba(Xte2)[:, 1])
    sint = _sintetiza(df, SEMILLA)
    Xs2, ys2 = _prep(sint)
    auc_sint = _auc(yte2, _HGBC(random_state=SEMILLA).fit(Xs2, ys2)
                    .predict_proba(Xte2)[:, 1])
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.bar(["datos\nreales", "sintético\ningenuo"], [auc_real, auc_sint],
           color=[AZUL, BERMELLON], width=0.6)
    ax.axhline(0.5, ls="--", color="0.6", lw=1)
    ax.text(0.5, 0.515, "azar", color="0.5", fontsize=8,
            ha="center", va="bottom")
    for i, v in enumerate([auc_real, auc_sint]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0.45, 0.85)
    ax.set_ylabel("AUC en el test real")
    ax.set_title("Utilidad: el sintético ingenuo la pierde")
    fig.tight_layout(); fig.savefig(FIG / "cap15_sintetico.pdf"); plt.close(fig)

    # cap15_entropia: la reidentificación medida en bits. (a) la escalera del
    # cuasi-identificador sobre un servicio de 150000 oyentes: la entropía
    # acumulada sube hasta el techo log2(N), que la huella de escucha satura, y la
    # huella real de navegador (~18 bits, Eckersley 2010) lo rebasa. (b) el
    # umbral de identificación log2(N) a distintas escalas de poblacion: los bits
    # que hacen falta para senalar a alguien.
    def _bits(d, quasi):
        p = d.groupby(quasi).size() / len(d)
        return float(-(p * np.log2(p)).sum())

    esc = [("demográfico", ["pais", "edad", "sexo"]),
           ("+ grafo\nsocial", ["pais", "edad", "sexo", "sigue_a"]),
           ("+ huella de\nescucha", ["pais", "edad", "sexo", "huella_top50"])]
    dd = generar(n=150_000, seed=42)
    dd["edad"] = dd["edad"] // 10 * 10
    bits = [_bits(dd, q) for _, q in esc]
    techo = float(np.log2(len(dd)))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.3))

    # (a) la escalera de bits sobre un servicio de 150000 oyentes
    xx = np.arange(len(esc))
    a1.bar(xx, bits, width=0.6, color=AZUL, zorder=3)
    for xi, v in zip(xx, bits):
        a1.text(xi, v + 0.4, f"{v:.1f}".replace(".", ","), ha="center",
                fontsize=8, color=AZUL)
    a1.axhline(techo, ls="--", lw=1.1, color=TINTA, alpha=0.8, zorder=2)
    a1.text(2.36, techo, f"techo\n{techo:.1f} bits".replace(".", ","),
            color=TINTA, fontsize=7.4, va="center", ha="left", linespacing=0.9)
    a1.axhline(18.1, ls=":", lw=1.3, color=BERMELLON, zorder=2)
    a1.text(-0.45, 18.35, "huella de navegador $\\approx$ 18 bits",
            color=BERMELLON, fontsize=7.4, va="bottom", ha="left")
    a1.set_xticks(xx); a1.set_xticklabels([e[0] for e in esc], fontsize=8)
    a1.set_ylabel("entropía acumulada (bits)")
    a1.set_ylim(0, 20.5); a1.set_xlim(-0.6, 3.05)
    a1.set_title("Un servicio de 150 000 oyentes", fontsize=9)

    # (b) el umbral log2(N) a distintas escalas de poblacion
    escalas = [("un pueblo", 130_000), ("una capital", 3_340_000),
               ("España", 48_600_000), ("el mundo", 8_200_000_000)]
    ng = np.logspace(2, 10, 400)
    a2.fill_between(ng, np.log2(ng), 36, color=AZUL, alpha=0.07, zorder=1)
    a2.plot(ng, np.log2(ng), color=TINTA, lw=1.8, zorder=3)
    a2.text(2e3, 30, "identificado", color="0.45", fontsize=8, ha="center",
            style="italic")
    a2.text(3e7, 6, "anónimo en el grupo", color="0.45", fontsize=8,
            ha="center", style="italic")
    for nombre, pob in escalas:
        yb = float(np.log2(pob))
        a2.scatter([pob], [yb], color=AZUL, s=30, zorder=5)
        a2.annotate(f"{nombre} ({yb:.1f})".replace(".", ","), (pob, yb),
                    textcoords="offset points", xytext=(-7, 6),
                    ha="right", va="bottom", fontsize=7.6, color=TINTA)
    a2.axhline(18.1, ls="--", lw=1.2, color=BERMELLON, zorder=4)
    a2.text(2e8, 18.6, "huella de navegador (18 bits)", color=BERMELLON,
            fontsize=7.4, va="bottom", ha="center")
    a2.set_xscale("log")
    a2.set_xlim(8e1, 6e10); a2.set_ylim(0, 36)
    a2.set_xlabel("usuarios (escala logarítmica)")
    a2.set_ylabel("bits necesarios ($\\log_2 N$)")
    a2.set_title("El umbral de identificación", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "cap15_entropia.pdf"); plt.close(fig)

    for nom in ["reident", "epsilon", "mia", "dp_hist", "sintetico", "entropia"]:
        print("generada:", (FIG / f"cap15_{nom}.pdf").name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Código del cap. 15 (privacidad).")
    ap.add_argument("--cifras", action="store_true")
    ap.add_argument("--figuras", action="store_true")
    a = ap.parse_args()
    if not (a.cifras or a.figuras):
        a.cifras = a.figuras = True
    if a.cifras:
        cifras()
    if a.figuras:
        figuras()
