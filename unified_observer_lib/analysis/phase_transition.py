import numpy as np
from scipy.stats import norm

class PhaseTransitionAnalyzer:
    def __init__(self, unified_observer):
        self.uo = unified_observer

    def detect_phase_transition(self, time_series, window_size=50, threshold=2):
        diff = np.diff(time_series)
        rolling_std = np.array([np.std(diff[i:i+window_size]) for i in range(len(diff)-window_size)])
        
        transition_points = np.where(rolling_std > threshold * np.mean(rolling_std))[0]
        return transition_points + window_size // 2

    def analyze_observer_transitions(self):
        q_history = [state['q'] for state in self.uo.observation_history]
        transition_points = self.detect_phase_transition(q_history)
        
        print(f"Detected {len(transition_points)} potential phase transitions")
        return transition_points

    def compute_order_parameter(self, data):
        return np.mean(data)

    def estimate_critical_exponents(self, data, control_parameter):
        order_parameter = self.compute_order_parameter(data)
        
        log_order = np.log(order_parameter)
        log_control = np.log(np.abs(control_parameter))
        
        slope, intercept = np.polyfit(log_control, log_order, 1)
        
        return -slope

    def generate_phase_diagram(self, parameter_range, num_points=100):
        parameters = np.linspace(parameter_range[0], parameter_range[1], num_points)
        order_parameters = []
        
        for param in parameters:
            self.uo.q = param
            data = self.uo.observe(np.random.randn())
            order_parameters.append(self.compute_order_parameter(data))
        
        return parameters, order_parameters
