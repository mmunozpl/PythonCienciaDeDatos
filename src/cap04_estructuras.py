"""Código del capítulo 4 — Estructuras de datos integradas y su coste.

Demuestra: el coste de 'in' en lista vs conjunto (reporta el COCIENTE, no el
tiempo absoluto), frecuencias con Counter, ahorro de memoria de un generador,
una dataclass frozen hashable, y el O(n^2) accidental vs un índice de conjunto.

Ejecutar:  uv run python src/cap04_estructuras.py
"""

from __future__ import annotations

import sys
import timeit
from collections import Counter
from dataclasses import dataclass


def cociente_pertenencia(n: int = 100_000) -> float:
    """Tiempo de 'x in lista' / 'x in conjunto' para un elemento ausente.

    Se reporta el cociente (adimensional, estable entre máquinas), no el
    tiempo absoluto, que depende del hardware (política de datos, cap. 10).
    """
    lista = list(range(n))
    conjunto = set(lista)
    ausente = -1
    t_lista = timeit.timeit(lambda: ausente in lista, number=200)
    t_conj = timeit.timeit(lambda: ausente in conjunto, number=200)
    return t_lista / t_conj


def frecuencias(texto: str, k: int = 3) -> list[tuple[str, int]]:
    """Las k palabras más frecuentes, con Counter."""
    return Counter(texto.split()).most_common(k)


@dataclass(frozen=True)
class Pista:
    """Registro inmutable y hashable: puede ser elemento de un set."""
    artista: int
    genero: str
    disco: int
    popularidad: float


def comunes_lento(a: list[int], b: list[int]) -> list[int]:
    """O(n*m): 'in lista' (O(m)) dentro del bucle."""
    return [x for x in a if x in b]


def comunes_rapido(a: list[int], b: list[int]) -> list[int]:
    """O(n+m): 'in set' (O(1)) tras convertir b a conjunto."""
    conjunto_b = set(b)
    return [x for x in a if x in conjunto_b]


if __name__ == "__main__":
    ratio = cociente_pertenencia()
    print(f"'in lista' es ~{ratio:.0f}x más lento que 'in conjunto' [medido]")
    if ratio < 5:
        raise RuntimeError(
            "cociente inesperadamente bajo: revisar la medición")

    print("frecuencias:", frecuencias("el dato el modelo el error"))

    pistas = {Pista(8, "pop", 3, 41.5),
              Pista(8, "pop", 3, 41.5)}                  # dos iguales -> uno
    print(f"pistas en el set (dos iguales colapsan): {len(pistas)}")

    a, b = list(range(1000)), list(range(500, 1500))
    assert comunes_lento(a, b) == comunes_rapido(a, b)   # mismo resultado
    print("comunes: lento y rápido dan el mismo resultado ✓")

    # ahorro de memoria: la lista materializa; el generador no
    lista = [x * x for x in range(100_000)]
    gen = (x * x for x in range(100_000))
    print(f"memoria lista={sys.getsizeof(lista)} B, "
          f"generador={sys.getsizeof(gen)} B")
