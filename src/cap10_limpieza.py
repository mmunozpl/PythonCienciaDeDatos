"""Código del capítulo 10 — Limpieza, calidad y transformación.

Reúne el código verificado del capítulo sobre el dataset transversal
del libro (catálogo musical de Spotify, cap. 5): la carga de la rebanada
de trabajo (seis géneros, clave natural única, tempo de 0 bpm como
lectura fallida), el generador de suciedad del § integrador (30 comas
decimales, 15 tempos imposibles, 10 centinelas, 20 erratas de género,
25 duplicados; semilla 42, índices disjuntos), la disciplina de tipos
(to_numeric revela lo no numérico), el contrato de esquema con pandera
(clave natural única, lecturas sin tempo válidas como NULL), la limpieza
con criterio (reparar / deduplicar / descartar solo lo imposible), la
revalidación y la transformación (banda de tempo, booleano de energía,
media por género). Si pandera no está instalado, la validación se omite
con aviso (degradación elegante); el resto del flujo se ejecuta igual.

Ejecutar:  uv run python src/cap10_limpieza.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PARQUET = Path("data/processed/musica.parquet")
CSV_SUCIO = Path("data/raw/musica_sucia.csv")
GENEROS = ["classical", "hip-hop", "jazz", "pop", "reggaeton", "rock"]
RANGO_TEMPO = (0.0, 250.0)  # bpm plausibles
CLAVE = ["track_id", "track_genre"]


def cargar_musica(ruta: Path = PARQUET) -> pd.DataFrame:
    """Lee el catálogo del cap. 5, se queda con la rebanada de seis
    géneros y la deduplica por clave natural. Un tempo de 0 bpm es una
    lectura fallida (cap. 5): se marca ausente."""
    df = (pd.read_parquet(ruta)
          .query("track_genre in @GENEROS")
          .drop_duplicates(subset=CLAVE)
          .reset_index(drop=True))
    tempo = df["tempo"].astype("float64").replace(0.0, np.nan)
    return df.assign(tempo=tempo, energy=df["energy"].astype("float64"))


def ensuciar(limpio: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Inyecta los defectos típicos de un fichero real (índices
    DISJUNTOS: cada fila recibe un defecto como mucho). Antes marca
    ausentes las lecturas que el analizador no resolvió (~2 %)."""
    df = limpio.copy()
    rng = np.random.default_rng(seed)             # lecturas sin pulso claro
    tempo = df["tempo"].to_numpy(copy=True)
    tempo[rng.random(len(df)) < 0.02] = np.nan
    df["tempo"] = tempo
    rng = np.random.default_rng(seed)             # las cinco plagas
    candidatas = rng.permutation(df.index[df["tempo"].notna()])
    coma, neg = candidatas[:30], candidatas[30:45]
    cent, dup = candidatas[45:55], candidatas[55:80]
    regg = [i for i in candidatas[80:]
            if df.at[i, "track_genre"] == "reggaeton"][:20]
    df["tempo"] = df["tempo"].astype(object)
    df.loc[coma, "tempo"] = [str(v).replace(".", ",")
                             for v in df.loc[coma, "tempo"]]
    df.loc[neg, "tempo"] = -1.0             # bpm imposible
    df.loc[cent, "tempo"] = 9999.0          # centinela sin documentar
    df.loc[regg, "track_genre"] = "regeton"  # errata: falta una g
    return pd.concat([df, df.loc[dup]], ignore_index=True)


def reparar_tipos(sucio: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Tipos primero: repara comas decimales y convierte con coerce."""
    como_num = pd.to_numeric(sucio["tempo"], errors="coerce")
    no_numericos = int((como_num.isna() & sucio["tempo"].notna()).sum())
    tempo = pd.to_numeric(
        sucio["tempo"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce")
    return sucio.assign(tempo=tempo), no_numericos


def informe_contrato(df: pd.DataFrame) -> pd.Series | None:
    """Valida con el contrato del capítulo; devuelve el parte agrupado
    (o None si pasa o si pandera no está instalado)."""
    try:
        import pandera.pandas as pa
    except ModuleNotFoundError:
        print("(pandera no disponible: se omite el contrato)")
        return None
    esquema = pa.DataFrameSchema(
        {
            "track_id": pa.Column(str),
            "track_genre": pa.Column(str, pa.Check.isin(GENEROS)),
            "popularity": pa.Column(int, pa.Check.in_range(0, 100),
                                    coerce=True),
            "energy": pa.Column(float, pa.Check.in_range(0.0, 1.0)),
            "tempo": pa.Column(float, pa.Check.in_range(*RANGO_TEMPO),
                               nullable=True),
        },
        unique=CLAVE,
    )
    try:
        esquema.validate(df, lazy=True)
        return None
    except pa.errors.SchemaErrors as err:
        return err.failure_cases["check"].value_counts()


def limpiar(tipado: pd.DataFrame) -> pd.DataFrame:
    """Repara lo reparable, deduplica por clave natural y descarta
    solo lo físicamente imposible; conserva las ausencias (NULL)."""
    return (
        tipado
        .assign(track_genre=lambda d:
                d["track_genre"].replace({"regeton": "reggaeton"}))
        .drop_duplicates(subset=CLAVE)
        .query("tempo.isna() or (0 <= tempo <= 250 and tempo != 9999)")
    )


def transformar(limpio: pd.DataFrame) -> pd.DataFrame:
    """Variables con dominio dentro: banda de tempo y booleano de energía."""
    return limpio.assign(
        energetica=lambda d: d["energy"] >= 0.5,
        banda=lambda d: pd.cut(d["tempo"], bins=[0, 90, 120, 150, 300],
                               labels=["balada", "medio", "movido",
                                       "rapido"]),
    )


if __name__ == "__main__":
    original = cargar_musica()
    print(f"origen: {PARQUET} | forma: {original.shape} | "
          f"sin tempo: {int(original['tempo'].isna().sum())}")

    # el artefacto sucio es un CSV, como el que descarga cualquiera
    CSV_SUCIO.parent.mkdir(parents=True, exist_ok=True)
    ensuciar(original).to_csv(CSV_SUCIO, sep=";", index=False)
    sucio = pd.read_csv(CSV_SUCIO, sep=";")
    print(f"ensuciado ({CSV_SUCIO}): {sucio.shape} (comas, imposibles, "
          "centinelas, erratas y duplicados)")

    tipado, no_num = reparar_tipos(sucio)
    print(f"no numericos revelados: {no_num} | sin tempo tras reparar: "
          f"{int(tipado['tempo'].isna().sum())}")

    parte = informe_contrato(tipado)
    if parte is not None:
        print(f"contrato: {int(parte.sum())} violaciones")
        print(parte.to_string())
        dups = int(tipado.duplicated(subset=CLAVE).sum())
        print(f"filas duplicadas reales: {dups}")

    limpio = limpiar(tipado)
    print(f"tras limpiar: {limpio.shape} | descartadas vs original: "
          f"{len(original) - len(limpio)}")
    if informe_contrato(limpio) is None:
        print("revalidacion: el contrato pasa")

    completo = transformar(limpio)
    por_genero = completo.groupby("track_genre")["energy"].mean()
    print(f"media de energy por genero ({len(por_genero)} generos), "
          f"media global = {completo['energy'].mean():.3f}")
    print(f"banda movido: {int((completo['banda'] == 'movido').sum())}"
          f" | energeticas: {int(completo['energetica'].sum())}")
