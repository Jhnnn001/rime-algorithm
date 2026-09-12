"""Box-bounded optimization with the four published RIME-family methods."""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from _base import Result
from acrime import ACRIME
from herime import HERIME
from qgrime import QGRIME
from rime import RIME


def minimize(objective: Callable[[np.ndarray], float], bounds: ArrayLike, method: str = "rime", **kwargs) -> Result:
    """Minimize an objective using a case-insensitive method name and constructor options."""
    methods = {"rime": RIME, "acrime": ACRIME, "qgrime": QGRIME, "herime": HERIME}
    if not isinstance(method, str) or method.lower() not in methods:
        raise ValueError(f"method must be one of {', '.join(methods)}.")
    return methods[method.lower()](objective, bounds, **kwargs).run()
