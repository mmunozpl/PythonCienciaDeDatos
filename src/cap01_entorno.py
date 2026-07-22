"""Código del capítulo 1 — Entorno de trabajo y flujo reproducible.

Demuestra tres ideas del capítulo con código ejecutable y determinista:
  1. leer la identidad del entorno (no teclear versiones a mano);
  2. la guarda ``__main__`` que separa "lo que hace" de "lo que ofrece";
  3. una comprobación mínima de determinismo (dos corridas deben cuadrar).

Ejecutar:  uv run python src/cap01_entorno.py
Es determinista: fija la semilla y no depende de la máquina salvo en las
versiones que reporta.
"""

from __future__ import annotations

import platform
import sys

import numpy as np


def identidad_entorno() -> dict[str, str]:
    """Devuelve la identidad del entorno: versiones que se deben registrar.

    Se leen del propio intérprete y de los paquetes, nunca se teclean, para que
    el informe de reproducibilidad no pueda quedar desactualizado.
    """
    return {
        "python": sys.version.split()[0],
        "implementacion": platform.python_implementation(),
        "numpy": np.__version__,
        "plataforma": platform.system(),
    }


def demo_determinismo(semilla: int = 2026, n: int = 5) -> np.ndarray:
    """Genera ``n`` valores con semilla fija; dos llamadas iguales cuadran.

    Es la versión mínima de la disciplina del capítulo: antes de citar
    una cifra aleatoria, se re-corre y se comprueba que coincide.
    """
    rng = np.random.default_rng(semilla)
    return rng.normal(size=n)


def _muestra(valores: np.ndarray, k: int = 5) -> list[float]:
    """Muestra hasta ``k`` observaciones redondeadas (inspección)."""
    return [round(float(v), 4) for v in valores[:k]]


if __name__ == "__main__":
    entorno = identidad_entorno()
    print("Entorno reproducible:")
    for clave, valor in entorno.items():
        print(f"  {clave:15s}: {valor}")

    a = demo_determinismo()
    b = demo_determinismo()
    cuadran = bool(np.array_equal(a, b))
    print("\nDeterminismo (dos corridas con misma semilla):")
    print(f"  muestra A: {_muestra(a)}")
    print(f"  muestra B: {_muestra(b)}")
    print(f"  ¿cuadran bit a bit?: {cuadran}")
    if not cuadran:  # validacion robusta con raise (no assert)
        raise RuntimeError(
            "determinismo roto: revisar la semilla del generador")
