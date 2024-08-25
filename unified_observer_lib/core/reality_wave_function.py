import numpy as np
import cupy as cp
import mpmath
from functools import lru_cache

class RealityWaveFunction:
    def __init__(self, unified_observer):
        self.uo = unified_observer
        self.xp = cp if unified_observer.use_gpu else np
        self.wave_history = []

    @lru_cache(maxsize=1000)
    def psi_R(self, q, tau):
        if tau == 0 or q == 0:
            return 1
        if q == 1:
            return 0
        
        n = self.xp.arange(1, 100)
        products = self.xp.prod(1 - q**n)
        return mpmath.power(q, 1/24) * products

    def evolve(self, delta_tau):
        new_tau = self.uo.tau + delta_tau
        new_psi = self.psi_R(self.uo.q, new_tau)
        self.wave_history.append(new_psi)
        return new_psi

    @lru_cache(maxsize=100)
    def quantum_entanglement(self, other_wave_function):
        return self.xp.dot(self.wave_history, other_wave_function.wave_history) / (
            self.xp.linalg.norm(self.wave_history) * self.xp.linalg.norm(other_wave_function.wave_history)
        )

    def phase_sensitive_evolution(self, delta_tau, phase_shift):
        new_tau = self.uo.tau + delta_tau
        new_psi = self.psi_R(self.uo.q, new_tau)
        phase_factor = mpmath.exp(1j * phase_shift)
        return new_psi * phase_factor
