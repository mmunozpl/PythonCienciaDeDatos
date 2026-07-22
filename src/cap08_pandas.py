"""Código del capítulo 8 — pandas: análisis tabular.

Trabaja sobre el dataset transversal del libro: el catálogo de música
(Spotify Tracks, maharshipandya), 113 999 pistas con rasgos de audio en
formato ANCHO ---una fila por pista, un rasgo por columna---. Reúne, en el
orden del capítulo, el código verificado: carga del Parquet limpio
(backend clásico y PyArrow) con comprobación de forma; la agregación
nombrada por género y el transform por género de la §8.4; el merge con el
catálogo de géneros en miniatura (left frente a inner, con validate);
pivot_table (tabla dinámica) y melt de los rasgos; la serie sintética de
reproducciones diarias de la §8.5 (resample y rolling) y la lección
honesta del integrador: la popularidad no se predice desde el sonido.

El Parquet lo produce datos_musica/construir_musica.py desde el crudo de
Hugging Face; no se genera dato sintético aquí.

Ejecutar:  uv run python src/cap08_pandas.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PARQUET = Path("data/processed/musica.parquet")
# el puñado de géneros con el que se comparan grupos a lo largo del capítulo
GENEROS = ["pop", "rock", "classical", "hip-hop", "jazz", "reggaeton"]
RASGOS = ["danceability", "energy", "valence", "tempo", "loudness",
          "acousticness", "speechiness", "instrumentalness",
          "liveness", "duration_ms"]


def cargar_musica(ruta: Path = PARQUET) -> pd.DataFrame:
    """Lee el Parquet limpio del catálogo (una fila por pista, cero nulos).

    No hay fallback sintético: el artefacto lo produce
    datos_musica/construir_musica.py desde el crudo de Hugging Face."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"no existe {ruta}; genéralo con datos_musica/construir_musica.py")
    musica = pd.read_parquet(ruta)
    # contrato del artefacto: 113 999 pistas, 20 columnas, sin ausentes.
    if musica.shape != (113999, 20):
        raise RuntimeError(f"forma inesperada del catálogo: {musica.shape}")
    return musica


def dtypes_pyarrow(ruta: Path = PARQUET) -> pd.Series | None:
    """Carga con dtype_backend="pyarrow" (§8.3): dtypes respaldados por
    Arrow, con las cadenas en un búfer UTF-8 contiguo (string[pyarrow])."""
    if not ruta.exists():
        return None
    return pd.read_parquet(ruta, dtype_backend="pyarrow").dtypes


def subconjunto(musica: pd.DataFrame) -> pd.DataFrame:
    """Las pistas de los seis géneros del puñado (6000 filas)."""
    return musica[musica["track_genre"].isin(GENEROS)].copy()


def resumen_por_genero(seis: pd.DataFrame) -> pd.DataFrame:
    """Split-apply-combine: media, mediana, recuento y pico de energy por
    género (energy es el rasgo continuo que resumimos, papel del valor)."""
    return seis.groupby("track_genre", observed=True).agg(
        media=("energy", "mean"),
        mediana=("energy", "median"),
        n=("energy", "count"),
        pico=("energy", "max"),
    ).round(3)


def sobre_su_mediana(seis: pd.DataFrame) -> pd.Series:
    """transform de la §8.4: compara la energy de cada pista con la mediana
    de SU género sin colapsar la tabla."""
    mediana = (seis.groupby("track_genre", observed=True)
               ["energy"].transform("median"))
    return seis["energy"] > mediana


def catalogo_generos() -> pd.DataFrame:
    """Catálogo de géneros en miniatura (§8.4); reggaeton queda fuera a
    propósito para ilustrar la diferencia entre left e inner."""
    return pd.DataFrame({
        "track_genre": ["pop", "rock", "classical", "hip-hop", "jazz"],
        "familia": ["mainstream", "banda", "clásica", "urbana", "clásica"],
        "vocal": [True, True, False, True, False],
    })


def unir_con_catalogo(seis: pd.DataFrame, generos: pd.DataFrame,
                      how: str = "left") -> pd.DataFrame:
    """Une pistas y catálogo de géneros declarando la relación esperada."""
    return seis.merge(generos, on="track_genre", how=how,
                      validate="many_to_one")


def largo_y_dinamica(seis: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """melt de los rasgos de audio a formato largo (una fila por
    pista-rasgo) y pivot_table como tabla dinámica género x explicit."""
    largo = (seis[["track_name", "energy", "danceability", "valence"]]
             .melt(id_vars="track_name", var_name="rasgo",
                   value_name="valor"))
    dinamica = seis.pivot_table(index="track_genre", columns="explicit",
                                values="popularity", observed=True)
    return largo, dinamica


def serie_reproducciones(seed: int = 2026) -> pd.Series:
    """Serie sintética de reproducciones diarias (en miles) de una pista a
    lo largo de un año: ilustra la maquinaria temporal de la §8.5, pues el
    catálogo no trae eje de tiempo. Con un registro de escuchas real
    ---fecha a fecha--- estas mismas operaciones aplican igual."""
    rng = np.random.default_rng(seed)             # semilla declarada
    fechas = pd.date_range("2026-01-01", periods=365, freq="D")
    return pd.Series(rng.normal(20, 5, 365), index=fechas)


def media_movil(serie: pd.Series, ventana: int = 7) -> pd.Series:
    """Media móvil de `ventana` días sobre la serie diaria (§8.5)."""
    return serie.rolling(ventana).mean()


def correlacion_popularidad(musica: pd.DataFrame) -> pd.Series:
    """La lección honesta del integrador: la popularidad no se predice desde
    los rasgos de audio (todos |r| <= 0,1). El éxito es social, no acústico."""
    return musica[["popularity"] + RASGOS].corr()["popularity"].drop(
        "popularity").round(3)


if __name__ == "__main__":
    musica = cargar_musica()
    print(f"forma: {musica.shape} | nulos: {int(musica.isna().sum().sum())}")

    tipos = dtypes_pyarrow()
    if tipos is not None:
        print("\ndtypes con dtype_backend='pyarrow' (selección):")
        print(tipos[["popularity", "energy", "tempo", "track_genre"]]
              .to_string())

    seis = subconjunto(musica)
    print(f"\nsubconjunto de los {len(GENEROS)} géneros: {seis.shape}")
    print("\nresumen de energy por género:")
    print(resumen_por_genero(seis))

    altas = sobre_su_mediana(seis)
    print(f"\nsobre la mediana de energy de su género: {altas.sum()}")

    generos = catalogo_generos()
    izq = unir_con_catalogo(seis, generos, how="left")
    inner = unir_con_catalogo(seis, generos, how="inner")
    print(f"\nmerge left: {len(izq)} filas "
          f"({izq['familia'].isna().sum()} sin ficha, reggaeton) | "
          f"inner: {len(inner)} filas")

    largo, dinamica = largo_y_dinamica(seis)
    print(f"\nmelt de rasgos: {largo.shape} | "
          f"pivot_table género x explicit: {dinamica.shape}")

    serie = serie_reproducciones()
    suave = media_movil(serie)
    print(f"\nserie diaria sintética: {serie.size} días; "
          f"media móvil 7d válida = {suave.dropna().size} valores")

    corr = correlacion_popularidad(musica)
    print("\ncorrelación de popularity con los rasgos (todos ~0):")
    print(corr.to_string())
