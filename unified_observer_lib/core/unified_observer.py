import numpy as np
import cupy as cp
import mpmath
from functools import lru_cache

class UnifiedObserver:
    def __init__(self, initial_q=0.5, initial_tau=0, use_gpu=False, precision=53):
        self.use_gpu = use_gpu
        self.xp = cp if use_gpu else np
        self.q = self.xp.array(initial_q)
        self.tau = self.xp.array(initial_tau)
        self.observation_history = []
        mpmath.mp.dps = precision

    def observe(self, data):
        self.observation_history.append(self.xp.array(data))
        self._update_parameters()

    def _update_parameters(self):
        if len(self.observation_history) > 1:
            recent_change = self.observation_history[-1] - self.observation_history[-2]
            self.q = self.xp.clip(self.q + recent_change * 0.01, 0, 1)
        self.tau += 1

    def get_state(self):
        return {"q": self.xp.asnumpy(self.q) if self.use_gpu else self.q,
                "tau": self.xp.asnumpy(self.tau) if self.use_gpu else self.tau}

    def adapt(self, feedback):
        self.q = self.xp.clip(self.q + feedback * 0.1, 0, 1)

    def reflect(self):
        """Method for self-reflection and adaptation"""
        avg_change = self.xp.mean(self.xp.diff(self.observation_history[-10:]))
        self.q = self.xp.clip(self.q + avg_change * 0.05, 0, 1)

    @lru_cache(maxsize=1000)
    def compute_complex_state(self):
        """Compute a complex state representation"""
        return mpmath.mpc(self.q, self.tau)
