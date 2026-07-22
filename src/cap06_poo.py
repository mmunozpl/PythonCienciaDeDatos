"""Código del capítulo 6 — POO y patrones para datos.

Demuestra: métodos especiales (Serie con __len__, __getitem__, __add__),
duck typing (evaluar funciona con cualquier objeto con la interfaz) y el
patrón
transformador fit/transform con la disciplina de NO aprender del test.

Ejecutar:  uv run python src/cap06_poo.py
"""

from __future__ import annotations

from collections.abc import Sequence


class Serie:
    """Secuencia numérica con operadores nativos vía métodos especiales."""

    def __init__(self, valores: list[float]) -> None:
        self._v = list(valores)

    def __len__(self) -> int:
        return len(self._v)

    def __getitem__(self, i):
        return self._v[i]

    def __repr__(self) -> str:
        return f"Serie({self._v!r})"

    def __eq__(self, otra: object) -> bool:
        return isinstance(otra, Serie) and self._v == otra._v

    def __add__(self, otra: Serie) -> Serie:
        return Serie([a + b for a, b in zip(self._v, otra._v, strict=True)])

    def __mul__(self, k: float) -> Serie:
        return Serie([a * k for a in self._v])


class MediaBase:
    """Modelo de referencia (duck typing): predice siempre la media de y."""

    def entrenar(self, y: Sequence[float]) -> MediaBase:
        self._media = sum(y) / len(y)
        return self

    def predecir(self, n: int) -> list[float]:
        return [self._media] * n


class EscaladorEstandar:
    """Transformador fit/transform: aprende SOLO en fit; aplica en transform."""

    def fit(self, x: Sequence[float]) -> EscaladorEstandar:
        self._media = sum(x) / len(x)
        var = sum((v - self._media) ** 2 for v in x) / len(x)
        self._desv = var ** 0.5 or 1.0
        return self

    def transform(self, x: Sequence[float]) -> list[float]:
        return [(v - self._media) / self._desv for v in x]


def evaluar(modelo, y: Sequence[float]) -> float:
    """Funciona con cualquier objeto con entrenar/predecir (duck
    typing)."""
    modelo.entrenar(y)
    pred = modelo.predecir(len(y))
    # error absoluto medio; strict=True detecta longitudes desparejadas
    return sum(abs(p - t) for p, t in zip(pred, y, strict=True)) / len(y)


if __name__ == "__main__":
    s = Serie([1.0, 2.0, 3.0])
    print("Serie:", len(s), s[0], s + Serie([10, 10, 10]), s * 2)
    assert (s * 2) == Serie([2.0, 4.0, 6.0])

    err = evaluar(MediaBase(), [2.0, 4.0, 6.0])
    print(f"MediaBase error absoluto medio: {err:.3f}")

    # fuga de datos: escalar aprendiendo del train, aplicando al test
    train, test = [10.0, 20.0, 30.0], [40.0, 50.0]
    esc = EscaladorEstandar().fit(train)          # aprende media/desv de TRAIN
    print("test escalado con parámetros de train:",
          [round(v, 3) for v in esc.transform(test)])
