from skopt import gp_minimize
from skopt.space import Real, Integer

class AdaptiveParameterOptimizer:
    def __init__(self, param_ranges):
        self.param_ranges = param_ranges

    def optimize(self, objective_function):
        result = gp_minimize(objective_function, self.param_ranges, n_calls=50, random_state=0)
        return result.x
