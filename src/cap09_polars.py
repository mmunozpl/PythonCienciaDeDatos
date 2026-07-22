"""Código del capítulo 9 — polars, Arrow y datos a gran escala.

Reúne el código verificado del capítulo sobre el catálogo de música del
libro: el Spotify Tracks Dataset limpio del cap. 5, real, 113 999 pistas
con 20 rasgos de audio y cero nulos. Contiene: la carga del artefacto
canónico con comprobaciones de recepción; el resumen perezoso de energy
por género con collect(engine="streaming") y su explain(); el destilado
de un género con sink_parquet; el particionado al estilo Hive por
track_genre (114 directorios) y la poda de particiones a la vista; el
JOIN de DuckDB con un catálogo de familias sonoras devuelto a polars; y
el puente final a NumPy. No hay generador sintético: el capítulo trabaja
sobre las 113 999 pistas reales; si el parquet no existe, remite al guion
que lo construye. Si polars/duckdb/pyarrow no están, lo indica y termina.

Ejecutar:  uv run python src/cap09_polars.py
"""

from __future__ import annotations

from pathlib import Path

PARQUET = Path("data/processed/musica.parquet")
CSV = Path("data/processed/musica.csv")
UN_GENERO = Path("data/processed/reggaeton.parquet")
POR_GENERO = Path("data/processed/musica_por_genero")

# el puñado de seis géneros con firma acústica reconocible; sirve para
# imprimir resúmenes de seis filas en vez de las 114 del catálogo.
SEIS = ["classical", "hip-hop", "jazz", "pop", "reggaeton", "rock"]

# catálogo de familias sonoras (la tabla de dimensión pequeña del JOIN):
# cada uno de los seis géneros pertenece a una familia por su sonido.
FAMILIAS = [
    ("classical", "acustica"),
    ("jazz", "acustica"),
    ("pop", "electrica"),
    ("rock", "electrica"),
    ("hip-hop", "urbana"),
    ("reggaeton", "urbana"),
]


def cargar_musica(parquet: Path):
    """Carga el catálogo canónico; sin fallback sintético."""
    import polars as pl

    if not parquet.exists():
        raise FileNotFoundError(
            f"falta {parquet}; reconstrúyelo con "
            "datos_musica/construir_musica.py")
    return pl.read_parquet(parquet)


def comprobar(parquet: Path) -> None:
    """Comprobaciones de recepcion: las cifras canonicas del capitulo."""
    df = cargar_musica(parquet)
    if df.shape != (113999, 20):
        raise ValueError(df.shape)
    if df.null_count().to_numpy().sum() != 0:
        raise ValueError("se esperaban cero nulos")
    if df["track_genre"].n_unique() != 114:
        raise ValueError("se esperaban 114 generos")


def resumen_por_genero(parquet: Path):
    """Plan perezoso + motor en flujo: media, n y pico de energy por
    genero, restringido al punado de seis para una salida compacta."""
    import polars as pl

    plan = (pl.scan_parquet(parquet)
            .filter(pl.col("track_genre").is_in(SEIS))
            .group_by("track_genre")
            .agg(media=pl.col("energy").mean().round(3),
                 n=pl.len(),
                 pico=pl.col("energy").max().round(3))
            .sort("track_genre"))
    return plan.collect(engine="streaming")


def plan_optimizado(parquet: Path) -> str:
    """El plan que ejecuta de verdad el optimizador (los dos empujes)."""
    import polars as pl

    plan = (pl.scan_parquet(parquet)
            .filter(pl.col("track_genre") == "classical")
            .select(pl.col("track_genre"), pl.col("energy"),
                    pl.col("acousticness")))
    return plan.explain()


def destilar_genero(parquet: Path, destino: Path) -> tuple[int, int]:
    """sink_parquet: filtra un genero y lo reescribe en flujo, sin RAM."""
    import polars as pl

    plan = (pl.scan_parquet(parquet)
            .filter(pl.col("track_genre") == "reggaeton"))
    plan.sink_parquet(destino)
    filas = pl.scan_parquet(destino).select(pl.len()).collect().item()
    return filas, destino.stat().st_size


def particionar_por_genero(parquet: Path, destino: Path) -> None:
    """Particionado al estilo Hive: un directorio por track_genre."""
    import polars as pl

    df = pl.read_parquet(parquet)
    df.write_parquet(destino, partition_by="track_genre")


def poda_de_un_genero(destino: Path):
    """La poda de particiones a la vista: solo se lee un directorio."""
    import polars as pl

    lf = (pl.scan_parquet(f"{destino.as_posix()}/**/*.parquet",
                          hive_partitioning=True)
          .filter(pl.col("track_genre") == "classical"))
    plan = lf.explain()
    filas = lf.select(pl.len()).collect().item()
    return plan, filas


def energia_por_familia(parquet: Path):
    """JOIN relacional en DuckDB (catalogo de familias) devuelto a polars.

    La firma acustica es real: la familia acustica (classical, jazz) suena
    muy por debajo de la urbana (hip-hop, reggaeton)."""
    import duckdb
    import polars as pl

    # duckdb lo lee por NOMBRE desde el SQL (replacement scan).
    familias = pl.DataFrame(  # noqa: F841
        FAMILIAS, schema=["track_genre", "familia"], orient="row"
    )
    return duckdb.sql(f"""
        SELECT f.familia,
               count(*)               AS n,
               round(avg(m.energy), 3) AS energy
        FROM read_parquet('{parquet.as_posix()}') AS m
        JOIN familias AS f USING (track_genre)
        GROUP BY f.familia
        ORDER BY energy DESC
    """).pl()


def main() -> None:
    try:
        import duckdb  # noqa: F401
        import polars as pl
        import pyarrow  # noqa: F401
    except ModuleNotFoundError as e:
        print(f"({e.name} no disponible); se omite el capítulo 9.")
        return

    comprobar(PARQUET)
    print(f"{PARQUET}: {PARQUET.stat().st_size / 1e6:.1f} MB "
          "(113999 pistas, 20 rasgos, 0 nulos, 114 generos)")
    if not CSV.exists():
        pl.read_parquet(PARQUET).write_csv(CSV)
    print(f"{CSV}: {CSV.stat().st_size / 1e6:.1f} MB (CSV gemelo)")

    print("\nresumen de energy por genero (lazy + motor en flujo):")
    print(resumen_por_genero(PARQUET))

    print("\nplan optimizado (pushdowns a la vista):")
    print(plan_optimizado(PARQUET))

    filas, tamano = destilar_genero(PARQUET, UN_GENERO)
    print(f"\nsink_parquet del reggaeton: {filas} filas, "
          f"{tamano / 1024:.0f} KB")

    if not POR_GENERO.exists():
        particionar_por_genero(PARQUET, POR_GENERO)
    plan, classical = poda_de_un_genero(POR_GENERO)
    print(f"\nparticionado por genero: classical = {classical} pistas; "
          "plan con poda:")
    print(plan)

    print("\nenergy media por familia (JOIN DuckDB -> polars):")
    familias = energia_por_familia(PARQUET)
    print(familias)

    # el puente final: de la tabla al array para scikit-learn (cap. 13).
    matriz = familias.select("energy", "n").to_numpy()
    print(f"\nto_numpy: matriz {matriz.shape} lista para el cap. 13")


if __name__ == "__main__":
    main()
