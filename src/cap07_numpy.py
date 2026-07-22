"""Código del capítulo 7 — NumPy: cómputo vectorizado.

Demuestra: el COCIENTE de velocidad vectorización vs bucle (no el tiempo
absoluto), centrado de columnas por broadcasting, conteo con máscara booleana
frente a la teoría, vista vs copia, y mínimos cuadrados por ecuación normal.

Ejecutar:  uv run python src/cap07_numpy.py
"""

from __future__ import annotations

import math
import timeit

import numpy as np


def cociente_vectorizacion(n: int = 1_000_000) -> float:
    """Tiempo del bucle Python / tiempo vectorizado para sqrt(x)*2+1.

    Se reporta el cociente (adimensional), no el tiempo absoluto (política de
    datos, cap. 10): la magnitud depende de la máquina, el orden no.
    """
    x = np.arange(n, dtype=np.float64)
    t_vec = timeit.timeit(lambda: np.sqrt(x) * 2 + 1, number=10)
    xs = list(range(n))
    t_bucle = timeit.timeit(
        lambda: [math.sqrt(v) * 2 + 1 for v in xs], number=10)
    return t_bucle / t_vec


def centrar_columnas(datos: np.ndarray) -> np.ndarray:
    """Resta la media de cada columna por broadcasting (n,d) - (d,)."""
    return datos - datos.mean(axis=0)


def contar_extremos(rng: np.random.Generator,
                    n: int = 100_000) -> tuple[float, float]:
    """Fracción de N(0,1) sobre 1.96 (máscara) vs teoría (~2,5 %)."""
    x = rng.normal(size=n)
    empirica = float((x > 1.96).mean())
    return empirica, 0.025


def minimos_cuadrados(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Coeficientes por ecuación normal, con columna de intercepto."""
    Xb = np.c_[np.ones(len(X)), X]
    return np.linalg.solve(Xb.T @ Xb, Xb.T @ y)


if __name__ == "__main__":
    ratio = cociente_vectorizacion()
    print(f"vectorización ~{ratio:.0f}x más rápida que el bucle [medido]")
    if ratio < 5:
        raise RuntimeError(
            "cociente inesperadamente bajo: revisar la medición")

    datos = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
    centrado = centrar_columnas(datos)
    print("medias tras centrar (~0):", np.round(centrado.mean(axis=0), 10))
    if not np.allclose(centrado.mean(axis=0), 0):
        raise RuntimeError("el centrado no dejó media ~0")

    rng = np.random.default_rng(2026)
    emp, teo = contar_extremos(rng)
    print(f"fracción > 1.96: empírica={emp:.4f}  teórica≈{teo}")

    # vista vs copia
    a = np.arange(10)
    vista = a[2:5]
    vista[0] = 999                            # modifica a
    copia = a[a > 100].copy()                 # no afecta a
    print(f"tras modificar la vista, a[2]={a[2]} (999, cambió)")

    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 6.0])
    coefs = np.round(minimos_cuadrados(X, y), 6)
    print("mínimos cuadrados (intercepto, pendiente):", coefs)
