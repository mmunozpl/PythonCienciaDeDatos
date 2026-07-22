"""Código del capítulo 11 — Estadística, probabilidad e inferencia.

Reúne el código verificado del capítulo sobre el catálogo de música (Spotify
Tracks, ya curado por el cap. 10): medidas resumen del tempo y el matiz de
ddof, la respuesta al porqué del valor bajo del cap. 10 (las vallas de Tukey
como 2,7 sigma), la ley de los grandes números y el teorema central del
límite por simulacion, el intervalo de confianza analitico y por bootstrap
(percentil y BCa de scipy), el contraste explicita/no explicita con Welch,
Cohen d y test de permutacion, y la simulacion de p-hacking con correccion
por comparaciones multiples. El catálogo entero (113 999 pistas) hace de
poblacion cuya verdad conocemos; sobre ella tomamos una muestra de trabajo
de 3 000 pistas con semilla 42; las simulaciones usan la semilla 2026.
Si scipy no esta instalado, las partes que lo usan se omiten con aviso.

Ejecutar:  uv run python src/cap11_estadistica.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PARQUET = Path("data/processed/musica.parquet")
SEMILLA_MUESTRA = 42       # la muestra de trabajo (papel de la semilla 42)
N_MUESTRA = 3_000          # escala manejable para el bootstrap (~230 MB)


def cargar_poblacion() -> pd.DataFrame:
    """El catálogo con tempo válido: descarta las 157 lecturas rotas (tempo 0),
    que son ausencias con firma de error, no mediciones. Es la población cuya
    verdad conocemos por tenerla entera."""
    if not PARQUET.exists():
        raise FileNotFoundError(f"no encuentro {PARQUET}")
    musica = pd.read_parquet(PARQUET)
    return musica[musica["tempo"] > 0]


def cargar_muestra(poblacion: pd.DataFrame) -> pd.DataFrame:
    """La muestra aleatoria de trabajo (semilla 42): 3 000 pistas del catálogo,
    una ventana con la que ver bailar una estimación."""
    return poblacion.sample(n=N_MUESTRA, random_state=SEMILLA_MUESTRA)


def resumen(x: np.ndarray) -> dict[str, float]:
    """Centro, dispersión y forma; ojo al ddof=1 (muestral) vs ddof=0."""
    q1, q3 = np.percentile(x, [25, 75])
    return {
        "n": len(x),
        "media": float(np.mean(x)),
        "mediana": float(np.median(x)),
        "desv_muestral": float(np.std(x, ddof=1)),   # pandas usa este
        "desv_poblacional": float(np.std(x)),        # numpy usa este
        "iqr": float(q3 - q1),
    }


def tukey_en_sigmas() -> float:
    """El porqué del valor bajo del cap. 10: en una normal las vallas de Tukey
    (1,5 IQR) caen en 2,698 sigma, mas estrictas que el 3 sigma."""
    from scipy import stats
    q3 = stats.norm.ppf(0.75)                # 0.6745 sigma
    return float(q3 + 1.5 * 2 * q3)          # valla superior en sigmas


def lgn(rng: np.random.Generator, n: int = 10_000) -> np.ndarray:
    """Ley de los grandes números: media acumulada de una exponencial."""
    x = rng.exponential(1.0, n)
    return np.cumsum(x) / np.arange(1, n + 1)


def tcl_asimetria(rng: np.random.Generator, n_muestra: int,
                  n_rep: int = 10_000) -> float:
    """Asimetría de n_rep medias de exponenciales de tamaño n_muestra."""
    from scipy import stats
    medias = rng.exponential(1.0, size=(n_rep, n_muestra)).mean(axis=1)
    return float(stats.skew(medias))


def ic_analitico(x: np.ndarray) -> tuple[float, float]:
    """IC 95 % de la media por el TCL: media +- 1,96 error estandar."""
    ee = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() - 1.96 * ee), float(x.mean() + 1.96 * ee)


def ic_bootstrap(x: np.ndarray, rng: np.random.Generator,
                 n_rep: int = 10_000) -> tuple[float, float]:
    """IC 95 % de la media por bootstrap percentil (con reemplazo)."""
    medias = rng.choice(x, size=(n_rep, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(medias, [2.5, 97.5])
    return float(lo), float(hi)


def contraste_explicit(muestra: pd.DataFrame) -> dict:
    """La pregunta del cap. 10: son mas rapidas las pistas explicitas?
    Welch, Cohen d y test de permutacion sobre el tempo de la muestra."""
    from scipy import stats
    exp = muestra.loc[muestra["explicit"], "tempo"].to_numpy()
    lim = muestra.loc[~muestra["explicit"], "tempo"].to_numpy()
    welch = stats.ttest_ind(exp, lim, equal_var=False)
    nx, ny = len(exp), len(lim)
    var = ((nx - 1) * exp.var(ddof=1) + (ny - 1) * lim.var(ddof=1))
    d = (exp.mean() - lim.mean()) / np.sqrt(var / (nx + ny - 2))

    def dif(a, b, axis=-1):
        return np.mean(a, axis=axis) - np.mean(b, axis=axis)

    perm = stats.permutation_test(
        (exp, lim), dif, permutation_type="independent",
        n_resamples=10_000, rng=np.random.default_rng(2026))
    return {"n_exp": nx, "n_lim": ny, "media_exp": exp.mean(),
            "media_lim": lim.mean(), "welch_p": welch.pvalue,
            "cohen_d": float(d), "perm_p": perm.pvalue}


def p_hacking(rng: np.random.Generator, n_vars: int = 100,
              n_obs: int = 200) -> dict:
    """Cuantas de n_vars variables SIN relacion dan p<0.05, y que queda
    tras controlar la tasa de falsos descubrimientos (Benjamini-Hochberg)."""
    from scipy import stats
    y = rng.normal(size=n_obs)
    ps = [stats.pearsonr(rng.normal(size=n_obs), y).pvalue
          for _ in range(n_vars)]
    ajustados = stats.false_discovery_control(ps)
    return {"crudos_sig": sum(p < 0.05 for p in ps),
            "menor_p": min(ps),
            "tras_bh": int(sum(a < 0.05 for a in ajustados)),
            "bonferroni": sum(p < 0.05 / n_vars for p in ps)}


def potencia(tempo: np.ndarray, n: int, delta: float,
             n_rep: int = 1000, seed: int = 2026) -> float:
    """Potencia de Welch por simulacion (ejercicio 10): dos muestras de
    tamano n del tempo de trabajo con reemplazo, sumando delta a la primera;
    proporcion de rechazos al 5 %. Semilla fija por combinacion: fija
    tambien el orden de los dos sorteos (parte del diseno)."""
    from scipy import stats
    rng = np.random.default_rng(seed)
    rechazos = 0
    for _ in range(n_rep):
        marca = rng.choice(tempo, size=n, replace=True) + delta
        resto = rng.choice(tempo, size=n, replace=True)
        rechazos += stats.ttest_ind(marca, resto, equal_var=False).pvalue < 0.05
    return rechazos / n_rep


if __name__ == "__main__":
    try:
        import scipy  # noqa: F401
    except ModuleNotFoundError:
        print("(scipy no disponible: se omiten inferencia y simulaciones)")
        raise SystemExit(0) from None

    poblacion = cargar_poblacion()
    mu_pob = float(poblacion["tempo"].mean())
    muestra = cargar_muestra(poblacion)
    tempo = muestra["tempo"].to_numpy()

    print(f"poblacion: {len(poblacion)} pistas con tempo valido, "
          f"media verdadera {mu_pob:.2f} BPM")
    r = resumen(tempo)
    print(f"tempo (muestra n={r['n']}): media={r['media']:.2f} "
          f"mediana={r['mediana']:.1f} s={r['desv_muestral']:.2f} "
          f"IQR={r['iqr']:.1f}")
    print(f"ddof: muestral {r['desv_muestral']:.4f} vs poblacional "
          f"{r['desv_poblacional']:.4f}")
    print(f"vallas de Tukey en una normal: {tukey_en_sigmas():.3f} sigma "
          "(mas estrictas que 3 sigma)")

    sim = np.random.default_rng(2026)
    ac = lgn(sim)
    print(f"LGN exponencial: media tras 10/100/10000 = "
          f"{ac[9]:.3f}/{ac[99]:.3f}/{ac[-1]:.4f} (esperanza 1)")
    sim = np.random.default_rng(2026)
    asim = [tcl_asimetria(sim, n) for n in (2, 5, 30)]
    print(f"TCL asimetria de las medias n=2/5/30: "
          f"{asim[0]:.2f}/{asim[1]:.2f}/{asim[2]:.2f}")

    print(f"IC 95 % media tempo: analitico "
          f"{tuple(round(v, 1) for v in ic_analitico(tempo))} | "
          f"bootstrap "
          f"{tuple(round(v, 1) for v in ic_bootstrap(tempo, np.random.default_rng(2026)))}")

    c = contraste_explicit(muestra)
    print(f"explicitas ({c['media_exp']:.2f}) vs no explicitas "
          f"({c['media_lim']:.2f}): Welch p={c['welch_p']:.3f}, "
          f"d={c['cohen_d']:.3f}, permutacion p={c['perm_p']:.3f}")

    h = p_hacking(np.random.default_rng(2026))
    print(f"p-hacking: {h['crudos_sig']}/100 con p<0.05 (menor "
          f"{h['menor_p']:.4f}); tras BH {h['tras_bh']}, "
          f"Bonferroni {h['bonferroni']}")

    # ejercicio 10: la curva de potencia (cifras ancladas al capitulo)
    print(f"potencia delta=0: {potencia(tempo, 50, 0.0):.3f} (n=50) / "
          f"{potencia(tempo, 200, 0.0):.3f} (n=200)")
    print(f"potencia delta=5: {potencia(tempo, 50, 5.0):.3f} / "
          f"{potencia(tempo, 200, 5.0):.3f}")
    print(f"potencia delta=10: {potencia(tempo, 50, 10.0):.3f} / "
          f"{potencia(tempo, 200, 10.0):.3f}")
