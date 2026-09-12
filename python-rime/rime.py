"""RIME: Su et al. (2023), doi:10.1016/j.neucom.2023.02.010."""

from _base import BaseRIME


class RIME(BaseRIME):
    """Minimize with the paper's soft rime, hard rime, and positive greedy selection."""

    method = "rime"
