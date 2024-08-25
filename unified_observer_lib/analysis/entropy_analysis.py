import numpy as np
from scipy.stats import entropy

class EntropyAnalyzer:
    def __init__(self, unified_observer):
        self.uo = unified_observer

    def compute_shannon_entropy(self, data):
        _, counts = np.unique(data, return_counts=True)
        return entropy(counts, base=2)

    def compute_tsallis_entropy(self, data, q=1.5):
        _, counts = np.unique(data, return_counts=True)
        probabilities = counts / np.sum(counts)
        return (1 - np.sum(probabilities**q)) / (q - 1)

    def analyze_observer_entropy(self):
        q_history = [state['q'] for state in self.uo.observation_history]
        shannon_entropy = self.compute_shannon_entropy(q_history)
        tsallis_entropy = self.compute_tsallis_entropy(q_history)
        
        return {
            'shannon_entropy': shannon_entropy,
            'tsallis_entropy': tsallis_entropy
        }

    def entropy_rate(self, data, lag=1):
        joint_entropy = self.compute_shannon_entropy(list(zip(data[:-lag], data[lag:])))
        marginal_entropy = self.compute_shannon_entropy(data[:-lag])
        return joint_entropy - marginal_entropy
