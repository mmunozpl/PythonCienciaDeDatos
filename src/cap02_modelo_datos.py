"""Código del capítulo 2 — El modelo de datos de Python.

Demuestra: identidad vs valor, referencias y mutabilidad, la trampa del
argumento mutable por defecto y su arreglo, el error de comparar floats con ==,
y caracteres vs bytes en Unicode. Todo determinista.

Ejecutar:  uv run python src/cap02_modelo_datos.py
"""

from __future__ import annotations

import math


def identidad_vs_valor() -> tuple[bool, bool]:
    """Dos listas iguales en valor no son el mismo objeto (== sí, is no)."""
    a = [1, 2, 3]
    b = [1, 2, 3]
    return (a == b, a is b)          # (True, False)


def agregar_mal(x: int, acc: list[int] = []) -> list[int]:  # noqa: B006
    """TRAMPA: el defecto mutable se crea una vez y acumula estado.

    El noqa B006 es deliberado: este es el antipatrón que el
    capítulo enseña; agregar_bien muestra la forma correcta.
    """
    acc.append(x)
    return acc


def agregar_bien(x: int, acc: list[int] | None = None) -> list[int]:
    """Arreglo idiomático: centinela None y creación dentro de la función."""
    acc = [] if acc is None else acc
    acc.append(x)
    return acc


def float_no_es_exacto() -> tuple[bool, bool]:
    """0.1 + 0.2 != 0.3 en coma flotante; isclose lo compara bien."""
    ingenuo = (0.1 + 0.2 == 0.3)             # False
    correcto = math.isclose(0.1 + 0.2, 0.3)  # True
    return (ingenuo, correcto)


def caracteres_vs_bytes(s: str) -> tuple[int, int]:
    """Longitud en puntos de código (caracteres) vs en bytes UTF-8."""
    return (len(s), len(s.encode("utf-8")))


if __name__ == "__main__":
    print("identidad vs valor (==, is):", identidad_vs_valor())

    # se imprime tras cada llamada: asi se ve el estado que acumula la
    # trampa. nota: las tres llamadas MAL devuelven EL MISMO objeto lista,
    # de modo que imprimirlas juntas al final mostraria tres veces [1, 2, 3]
    print("\nargumento mutable por defecto:")
    print("  MAL :", agregar_mal(1))   # [1]
    print("        ", agregar_mal(2))  # [1, 2]  la lista persiste
    print("        ", agregar_mal(3))  # [1, 2, 3]
    print("  BIEN:", agregar_bien(1))  # [1]
    print("        ", agregar_bien(2))  # [2]  instancia fresca por llamada
    print("        ", agregar_bien(3))  # [3]

    ingenuo, correcto = float_no_es_exacto()
    print(f"\nfloat: (0.1+0.2==0.3)={ingenuo}  isclose={correcto}")
    if ingenuo or not correcto:
        raise RuntimeError("comportamiento de coma flotante inesperado")

    s = "café ☕ 数据"
    chars, bytes_ = caracteres_vs_bytes(s)
    print(f"\nUnicode {s!r}: {chars} caracteres, {bytes_} bytes UTF-8")
