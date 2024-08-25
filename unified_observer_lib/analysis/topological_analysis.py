import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
from sklearn.metrics.pairwise import pairwise_distances

class TopologicalAnalyzer:
    def __init__(self, max_dimension=1):
        self.max_dimension = max_dimension

    def compute_persistence_diagrams(self, data):
        """Compute persistence diagrams for the given data"""
        diagrams = ripser(data, maxdim=self.max_dimension)['dgms']
        return diagrams

    def plot_persistence_diagrams(self, diagrams):
        """Plot the persistence diagrams"""
        plot_diagrams(diagrams, show=True)

    def compute_betti_numbers(self, diagrams):
        """Compute Betti numbers from persistence diagrams"""
        betti_numbers = []
        for i, diagram in enumerate(diagrams):
            betti_numbers.append(np.sum(np.isinf(diagram[:, 1])))
        return betti_numbers

    def analyze_time_series(self, time_series, embedding_dimension=2, delay=1):
        """Perform topological analysis on a time series"""
        N = len(time_series)
        embedded_data = np.array([time_series[i:i+embedding_dimension] 
                                  for i in range(0, N - embedding_dimension + 1, delay)])
        
        diagrams = self.compute_persistence_diagrams(embedded_data)
        betti_numbers = self.compute_betti_numbers(diagrams)
        
        return {
            'diagrams': diagrams,
            'betti_numbers': betti_numbers
        }

    def compare_topologies(self, data1, data2):
        """Compare topologies of two datasets using bottleneck distance"""
        diagrams1 = self.compute_persistence_diagrams(data1)
        diagrams2 = self.compute_persistence_diagrams(data2)
        
        bottleneck_distances = []
        for d in range(self.max_dimension + 1):
            distance = persim.bottleneck(diagrams1[d], diagrams2[d])
            bottleneck_distances.append(distance)
        
        return bottleneck_distances

    def analyze_observer_topology(self, unified_observer):
        """Analyze the topology of the observer's history"""
        q_history = [state['q'] for state in unified_observer.observation_history]
        tau_history = [state['tau'] for state in unified_observer.observation_history]
        
        data = np.column_stack((q_history, tau_history))
        result = self.analyze_time_series(data)
        
        print("Topological Analysis of Observer History:")
        print(f"Betti numbers: {result['betti_numbers']}")
        
        self.plot_persistence_diagrams(result['diagrams'])
        
        return result

    def compute_topological_complexity(self, data):
        """Compute topological complexity based on persistent homology"""
        diagrams = self.compute_persistence_diagrams(data)
        total_persistence = sum(np.sum(diagram[:, 1] - diagram[:, 0]) for diagram in diagrams)
        return total_persistence
