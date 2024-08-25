import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

class CatastropheAnalyzer:
    def __init__(self, unified_observer):
        self.uo = unified_observer

    def cusp_catastrophe(self, x, a, b):
        """
        Cusp catastrophe function
        V(x) = 1/4 * x^4 + 1/2 * a * x^2 + b * x
        """
        return 0.25 * x**4 + 0.5 * a * x**2 + b * x

    def find_equilibria(self, a, b):
        """Find equilibrium points of the cusp catastrophe"""
        def equilibrium(x):
            return x**3 + a * x + b
        
        roots = fsolve(equilibrium, [0, 1, -1])
        return roots[np.abs(equilibrium(roots)) < 1e-6]

    def analyze_catastrophe(self, a_range, b_range):
        """Analyze the catastrophe over a range of control parameters"""
        a_values = np.linspace(*a_range, 100)
        b_values = np.linspace(*b_range, 100)
        A, B = np.meshgrid(a_values, b_values)
        
        equilibria = np.array([[self.find_equilibria(a, b) for a in a_values] for b in b_values])
        num_equilibria = np.vectorize(len)(equilibria)
        
        return A, B, num_equilibria

    def plot_catastrophe(self, a_range, b_range):
        """Plot the catastrophe surface"""
        A, B, num_equilibria = self.analyze_catastrophe(a_range, b_range)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(A, B, num_equilibria, cmap='viridis')
        
        ax.set_xlabel('Control parameter a')
        ax.set_ylabel('Control parameter b')
        ax.set_zlabel('Number of equilibria')
        ax.set_title('Cusp Catastrophe Analysis')
        
        plt.colorbar(surf)
        plt.show()

    def detect_catastrophe(self, time_series, window_size=50, threshold=2):
        """Detect potential catastrophes in a time series"""
        diff = np.diff(time_series)
        rolling_std = np.array([np.std(diff[i:i+window_size]) for i in range(len(diff)-window_size)])
        
        catastrophe_points = np.where(rolling_std > threshold * np.mean(rolling_std))[0]
        return catastrophe_points + window_size // 2  # Adjust for window size

    def analyze_observer_catastrophes(self):
        """Analyze potential catastrophes in the observer's history"""
        q_history = [state['q'] for state in self.uo.observation_history]
        catastrophe_points = self.detect_catastrophe(q_history)
        
        print(f"Detected {len(catastrophe_points)} potential catastrophes")
        return catastrophe_points
