"""Small compatibility shim.

When Gymnasium is installed, HoolaEnv is a real gymnasium.Env and uses real
spaces.  The lightweight fallback keeps the core project/test suite runnable
without forcing Gymnasium on users who only want the console game.
"""
from __future__ import annotations

try:
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
    GYMNASIUM_AVAILABLE = True
    EnvBase = gym.Env
except ImportError:  # pragma: no cover - exercised only on minimal installs
    GYMNASIUM_AVAILABLE = False

    class EnvBase:
        metadata: dict = {}

        def reset(self, *, seed=None, options=None):
            return None

    class _Discrete:
        def __init__(self, n: int):
            self.n = int(n)

        def contains(self, x) -> bool:
            try:
                i = int(x)
            except Exception:
                return False
            return 0 <= i < self.n

    class _Box:
        def __init__(self, low, high, shape, dtype):
            self.low = low
            self.high = high
            self.shape = tuple(shape)
            self.dtype = dtype

    class _Dict:
        def __init__(self, spaces_dict):
            self.spaces = spaces_dict

    class _Spaces:
        Discrete = _Discrete
        Box = _Box
        Dict = _Dict

    spaces = _Spaces()
