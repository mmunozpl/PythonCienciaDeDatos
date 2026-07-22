"""Código del capítulo 5 — Entrada/salida, ficheros y formatos.

Demuestra: pathlib, ida y vuelta CSV con csv.DictReader, ida y vuelta JSON,
SQLite con consulta PARAMETRIZADA (nunca f-strings en SQL), y CSV -> Parquet con
el cociente de tamaños. Genera además dos CSV de apoyo REPRODUCIBLES para los
ejemplos de codificación y separadores del capítulo: generos.csv (acentos, con
copia en latin-1 para el mojibake) y europeo.csv (sep=";", coma decimal).

Mini-dataset [muestra] con la forma del real: pistas del Spotify Tracks
Dataset (género, popularidad 0-100, un rasgo de audio).

Ejecutar:  uv run python src/cap05_io_formatos.py
"""

from __future__ import annotations

import csv
import json
import sqlite3
import tempfile
from pathlib import Path

# muestra de pistas con la forma del dataset real (§5.10).
FILAS = [
    {"track_genre": "pop", "popularity": 90, "energy": 0.80},
    {"track_genre": "rock", "popularity": 55, "energy": 0.92},
    {"track_genre": "classical", "popularity": 40, "energy": 0.11},
]

# CSV de apoyo del capítulo: se dejan en data/ejemplos_cap05/ de forma
# reproducible (los usan los ejemplos de codificación y separadores).
EJEMPLOS = Path("data/ejemplos_cap05")


def escribir_csv(ruta: Path, filas: list[dict]) -> None:
    with ruta.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)


def leer_csv(ruta: Path) -> list[dict]:
    # el CSV guarda TODO como texto: hay que recastear los campos numéricos
    # (justamente el defecto que Parquet evita, §5.5).
    with ruta.open(encoding="utf-8", newline="") as f:
        return [{**fila, "popularity": int(fila["popularity"]),
                 "energy": float(fila["energy"])}
                for fila in csv.DictReader(f)]


def json_ida_vuelta(obj: dict) -> dict:
    return json.loads(json.dumps(obj, ensure_ascii=False))


def sqlite_consulta(filas: list[dict], umbral: int) -> list[tuple]:
    """Filtra por SQL con parámetros (?), nunca interpolando con f-strings."""
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE pistas(track_genre TEXT, popularity INT, "
                "energy REAL)")
    con.executemany(
        "INSERT INTO pistas VALUES (?,?,?)",
        [(f["track_genre"], f["popularity"], f["energy"]) for f in filas])
    filas_res = con.execute(
        "SELECT track_genre, popularity FROM pistas WHERE popularity > ? "
        "ORDER BY popularity DESC",
        (umbral,)).fetchall()
    con.close()
    return filas_res


def cociente_csv_parquet(ruta_csv: Path) -> float | None:
    """Tamaño CSV / tamaño Parquet (Parquet suele ocupar mucho menos)."""
    try:
        import pandas as pd
    except ModuleNotFoundError:
        return None
    df = pd.read_csv(ruta_csv)
    ruta_pq = ruta_csv.with_suffix(".parquet")
    df.to_parquet(ruta_pq)
    return ruta_csv.stat().st_size / ruta_pq.stat().st_size


def escribir_generos(destino: Path) -> tuple[str, str]:
    """CSV de géneros con acentos: se guarda una copia utf-8 y otra latin-1.
    Leer los bytes utf-8 con el códec latin-1 produce mojibake sin lanzar
    (latin-1 es una biyección total byte<->punto de código, §5.2)."""
    filas = [["id", "genero"], ["1", "clásica"], ["2", "electrónica"]]
    texto = "\n".join(";".join(f) for f in filas) + "\n"
    lat1 = destino.with_name("generos_latin1.csv")
    destino.write_text(texto, encoding="utf-8")
    lat1.write_text(texto, encoding="latin-1")
    bien = destino.read_text(encoding="utf-8").splitlines()[1]
    mal = destino.read_text(encoding="latin-1").splitlines()[1]
    return bien, mal


def escribir_europeo(destino: Path) -> list[dict]:
    """CSV «a la europea»: sep=";" y coma decimal. Se lee fijando el
    delimiter y normalizando la coma decimal antes de convertir (§5.3)."""
    contenido = (
        "genero;popularidad;energia;tempo\n"
        "pop;90;0,80;120,5\n"
        "classical;40;0,11;76,3\n"
    )
    destino.write_text(contenido, encoding="utf-8")
    registros: list[dict] = []
    with destino.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            registros.append({
                "genero": r["genero"],
                "popularidad": int(r["popularidad"]),
                "energia": float(r["energia"].replace(",", ".")),
                "tempo": float(r["tempo"].replace(",", ".")),
            })
    return registros


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        csv_ruta = base / "musica_muestra.csv"
        escribir_csv(csv_ruta, FILAS)
        recuperado = leer_csv(csv_ruta)
        print("CSV ida y vuelta ok:", recuperado == FILAS)

        obj = {"modelo": "rf", "hiperparametros": {"arboles": 200}}
        print("JSON ida y vuelta ok:", json_ida_vuelta(obj) == obj)

        print("SQLite (popularity>50):", sqlite_consulta(FILAS, 50))

        ratio = cociente_csv_parquet(csv_ruta)
        if ratio is not None:
            print(f"tamaño CSV/Parquet = {ratio:.2f}x "
                  "[medido, depende del dato]")
        else:
            print("(pandas/pyarrow no disponibles: se omite CSV->Parquet)")

    # CSV de apoyo persistentes y reproducibles.
    EJEMPLOS.mkdir(parents=True, exist_ok=True)
    bien, mal = escribir_generos(EJEMPLOS / "generos.csv")
    print("generos.csv utf-8 ->", bien, "| leído como latin-1 ->", mal)
    europeo = escribir_europeo(EJEMPLOS / "europeo.csv")
    print("europeo.csv (sep=';', coma decimal):", europeo)
    print("CSV de apoyo en:", EJEMPLOS.resolve())
