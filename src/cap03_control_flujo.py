"""Código del capítulo 3 — Control de flujo, funciones y manejo de errores.

Demuestra: match/case, iteración idiomática (enumerate/zip), una función con
argumentos keyword-only (resumir), un cierre (contador) y captura de excepciones
específicas. Determinista.

Ejecutar:  uv run python src/cap03_control_flujo.py
"""

from __future__ import annotations

from collections.abc import Callable, Sequence


def describir_punto(punto: tuple) -> str:
    """Desestructura una tupla con match/case (Python 3.10+)."""
    match punto:
        case (0, 0):
            return "origen"
        case (0, y):
            return f"eje Y en {y}"
        case (x, 0):
            return f"eje X en {x}"
        case (x, y):
            return f"punto ({x}, {y})"
        case _:
            return "no es un par de coordenadas"


def resumir(datos: Sequence[float], *, metodo: str = "media",
            decimales: int = 2) -> float:
    """Resume 'datos' con el método indicado. '*' fuerza paso por nombre."""
    if not datos:
        raise ValueError("no se puede resumir una secuencia vacía")
    if metodo == "media":
        valor = sum(datos) / len(datos)
    elif metodo == "mediana":
        xs = sorted(datos)
        n = len(xs)
        medio = n // 2
        valor = xs[medio] if n % 2 else (xs[medio - 1] + xs[medio]) / 2
    else:
        raise ValueError(f"método desconocido: {metodo!r}")
    return round(valor, decimales)


def contador() -> Callable[[], int]:
    """Devuelve una función que recuerda cuántas veces se la ha llamado.

    El estado (n) vive capturado en el cierre, no en una variable global.
    """
    n = 0

    def incrementar() -> int:
        nonlocal n
        n += 1
        return n

    return incrementar


def leer_entero(texto: str) -> int | None:
    """Convierte a entero; devuelve None solo ante el error esperado."""
    try:
        return int(texto)
    except ValueError:
        return None


if __name__ == "__main__":
    for p in [(0, 0), (0, 5), (3, 0), (2, 4), "no"]:
        print(f"  {p!r:10s} -> {describir_punto(p)}")

    xs = [2, 4, 4, 4, 5, 7, 9]
    mediana = resumir(xs, metodo="mediana")
    print(f"\nresumir {xs}: media={resumir(xs)}  mediana={mediana}")

    c = contador()
    print("cierre contador:", c(), c(), c())         # 1 2 3
    if [c() for _ in range(0)] != []:
        raise RuntimeError("el cierre no debería avanzar sin llamadas")

    print("leer_entero:", leer_entero("42"), leer_entero("x"))   # 42 None
