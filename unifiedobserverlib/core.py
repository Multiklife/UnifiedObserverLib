import numpy as np
import mpmath

class UnifiedObserver:
    def __init__(self, q, tau):
        self.q = self.check_q(q)
        self.tau = self.check_tau(tau)
        self.entropy = 0

    def update_parameters(self, q, tau):
        self.q = self.check_q(q)
        self.tau = self.check_tau(tau)
        self.update_entropy()

    @staticmethod
    def check_q(q):
        if q <= 0 or q >= 1:
            raise ValueError("q must be in the range (0, 1).")
        return q

    @staticmethod
    def check_tau(tau):
        if tau < 0:
            raise ValueError("tau must be non-negative.")
        return tau

    def update_entropy(self):
        self.entropy = -self.q * np.log(self.q) - (1 - self.q) * np.log(1 - self.q)

class RealityWaveFunction:
    def __init__(self, unified_observer):
        self.uo = unified_observer

    def psi_R(self, q, tau):
        q = self.uo.check_q(q)
        tau = self.uo.check_tau(tau)
        if tau == 0 or q == 0:
            return 1
        if q == 1:
            return 0
        return mpmath.qp(q, 24)

    def evolve(self, delta_tau):
        new_tau = self.uo.tau + delta_tau
        return self.psi_R(self.uo.q, new_tau)
