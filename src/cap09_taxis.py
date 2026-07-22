"""cap09_taxis.py -- Una medicion a escala real: los viajes de taxi de NYC.

La maqueta de musica (8,7 MB) sirve para *aprender* la maquinaria de flujo y
particiones del capitulo 9, pero es demasiado pequena para *medir* nada: por
ambas vias el tiempo es indistinguible y el techo de memoria no se nota. Este
guion corre la MISMA maquinaria sobre datos que ya no caben tan holgados --un
ano de viajes de taxi amarillo de Nueva York, datos publicos del NYC TLC,
nativos en Parquet y por meses-- y mide lo que la maqueta no podia ensenar:

  1. el techo de memoria del motor en flujo frente al motor en memoria, y
  2. la poda de particiones a escala real.

Los datos NO se versionan (varios cientos de MB); este guion los regenera.
Fuente: NYC Taxi & Limousine Commission, Trip Record Data (dominio publico).
Verificado con polars 1.42.1 sobre CPython 3.11.

Uso:
    python src/cap09_taxis.py            # descarga (si falta) y mide todo
    python src/cap09_taxis.py __eager    # uso interno (subproceso aislado)
    python src/cap09_taxis.py __stream   # uso interno (subproceso aislado)
"""
from __future__ import annotations

import resource
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import polars as pl

BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data"
MESES = [f"2024-{m:02d}" for m in range(1, 13)]
DIR = Path("data/nyc_taxi")
GLOB = str(DIR / "yellow_tripdata_2024-*.parquet")
PART = DIR / "por_mes"


def descargar() -> None:
    """Baja los doce Parquet mensuales si aun no estan en disco."""
    DIR.mkdir(parents=True, exist_ok=True)
    for mes in MESES:
        dst = DIR / f"yellow_tripdata_{mes}.parquet"
        if dst.exists():
            continue
        url = f"{BASE}/yellow_tripdata_{mes}.parquet"
        print(f"  descargando {mes} ...", flush=True)
        urllib.request.urlretrieve(url, dst)


def pico_ram_mb() -> float:
    """Pico de memoria residente del proceso (ru_maxrss, KiB en Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# --- La consulta unica que compararemos por ambas vias -----------------------
# Gasto medio y numero de viajes por numero de pasajeros, sobre el ano entero.
COLS = ["passenger_count", "total_amount"]


def _leer_ano_entero() -> pl.DataFrame:
    """Carga el ano completo, todas las columnas, en un solo DataFrame.

    Unifica de paso la deriva de esquema del TLC: las columnas de fecha
    vienen en microsegundos unos meses y en nanosegundos otros, y un
    read_parquet ingenuo del glob completo fallaria al concatenarlas.
    """
    fechas = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
    trozos = [pl.scan_parquet(f).with_columns(
                  [pl.col(c).cast(pl.Datetime("ns")) for c in fechas])
              for f in sorted(DIR.glob("yellow_tripdata_2024-*.parquet"))]
    return pl.concat(trozos).collect()               # el ano entero en RAM


def _consulta_eager() -> pl.DataFrame:
    df = _leer_ano_entero()                          # naive: todo a la RAM
    return (df.group_by("passenger_count")
              .agg(gasto=pl.col("total_amount").mean(), n=pl.len())
              .sort("passenger_count"))


def _consulta_stream() -> pl.DataFrame:
    return (pl.scan_parquet(GLOB)                    # perezoso + en flujo
              .group_by("passenger_count")
              .agg(gasto=pl.col("total_amount").mean(), n=pl.len())
              .sort("passenger_count")
              .collect(engine="streaming"))


def _medir(fn) -> tuple[pl.DataFrame, float]:
    t = time.perf_counter()
    res = fn()
    return res, time.perf_counter() - t


def _modo_aislado(nombre: str) -> None:
    """Ejecuta una via en un proceso limpio e imprime tiempo y pico de RAM."""
    fn = _consulta_eager if nombre == "__eager" else _consulta_stream
    res, dt = _medir(fn)
    print(f"{nombre}\t{res.height}\t{dt * 1000:.0f}\t{pico_ram_mb():.0f}")


def _subproceso(modo: str) -> tuple[int, float, float]:
    """Lanza este mismo guion en modo aislado y recoge (filas, ms, picoRAM)."""
    out = subprocess.run([sys.executable, __file__, modo],
                         capture_output=True, text=True, check=True).stdout
    linea = [l for l in out.splitlines() if l.startswith(modo)][0]
    _, filas, ms, ram = linea.split("\t")
    return int(filas), float(ms), float(ram)


def informe() -> None:
    # --- Hechos del conjunto -------------------------------------------------
    ficheros = sorted(DIR.glob("yellow_tripdata_2024-*.parquet"))
    tam_mb = sum(f.stat().st_size for f in ficheros) / 1024**2
    filas = pl.scan_parquet(GLOB).select(pl.len()).collect().item()
    print("== Hechos ==")
    print(f"ficheros={len(ficheros)}  filas={filas:,}  disco={tam_mb:.0f} MB")

    # --- 1. Techo de memoria: en memoria vs en flujo -------------------------
    fe, me, rame = _subproceso("__eager")
    fs, ms, rams = _subproceso("__stream")
    print("\n== Techo de memoria (misma consulta, ano entero) ==")
    print(f"en memoria (read_parquet):  {me:5.0f} ms   pico {rame:6.0f} MB")
    print(f"en flujo   (streaming):     {ms:5.0f} ms   pico {rams:6.0f} MB")
    print(f"ratio de memoria: x{rame / rams:.1f} menos en flujo")

    # --- 2. Poda de particiones a escala real --------------------------------
    if not PART.exists():
        print("\n(particionando por mes, una sola vez ...)", flush=True)
        # Cada fichero ES un mes: etiquetamos por su nombre, sin tocar fechas
        # (las columnas datetime del TLC mezclan unidades us/ns entre meses).
        trozos = []
        for mes in MESES:
            f = DIR / f"yellow_tripdata_{mes}.parquet"
            trozos.append(pl.read_parquet(f, columns=COLS)
                            .with_columns(mes=pl.lit(mes)))
        pl.concat(trozos).write_parquet(PART, partition_by="mes")

    glob_part = str(PART / "**/*.parquet")
    lf = (pl.scan_parquet(glob_part, hive_partitioning=True)
            .filter(pl.col("mes") == "2024-03")
            .select(gasto=pl.col("total_amount").mean(), n=pl.len()))
    plan = lf.explain()
    escaneados = plan.count("yellow") if "yellow" in plan else plan.count(".parquet")
    _, tp = _medir(lf.collect)
    print("\n== Poda de particiones (filtrar un mes de doce) ==")
    print(f"particiones leidas (SCAN): {escaneados} de 12")
    print(f"tiempo consulta de un mes: {tp * 1000:.0f} ms")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1].startswith("__"):
        _modo_aislado(sys.argv[1])
        return
    print("Preparando el ano de viajes de taxi de NYC (TLC) ...")
    descargar()
    informe()


if __name__ == "__main__":
    main()
