"""Standard continuous benchmarks with their conventional search boxes."""

import numpy as np


def sphere(x: np.ndarray) -> float:
    """Return the sum of squared coordinates, minimized at the origin."""
    return float(np.sum(np.square(x)))


def rastrigin(x: np.ndarray) -> float:
    """Return the Rastrigin value, minimized at the origin."""
    x = np.asarray(x)
    return float(10 * x.size + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def schwefel(x: np.ndarray) -> float:
    """Return the Schwefel value with the conventional rounded 418.9829 offset."""
    x = np.asarray(x)
    return float(418.9829 * x.size - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def zakharov(x: np.ndarray) -> float:
    """Return the Zakharov value with one-based coordinate weights."""
    x = np.asarray(x)
    weighted_sum = np.dot(0.5 * np.arange(1, x.size + 1), x)
    return float(np.sum(x**2) + weighted_sum**2 + weighted_sum**4)


def bounds(name: str, dim: int) -> np.ndarray:
    """Return the standard box as a ``(dim, 2)`` array for a named benchmark."""
    boxes = {
        "sphere": (-100, 100),
        "rastrigin": (-5.12, 5.12),
        "schwefel": (-500, 500),
        "zakharov": (-5, 10),
    }
    if not isinstance(name, str) or name.lower() not in boxes:
        raise ValueError("Unknown benchmark name.")
    if isinstance(dim, (bool, np.bool_)) or not isinstance(dim, (int, np.integer)) or dim < 1:
        raise ValueError("dim must be an integer >= 1.")
    return np.tile(boxes[name.lower()], (dim, 1)).astype(float)
