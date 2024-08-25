import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots

class AdvancedVisualizer:
    def __init__(self, unified_observer):
        self.uo = unified_observer

    def plot_parameter_evolution(self, num_steps):
        q_values = []
        tau_values = []
        for _ in range(num_steps):
            self.uo.observe(np.random.randn())
            state = self.uo.get_state()
            q_values.append(state['q'])
            tau_values.append(state['tau'])
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=("q parameter evolution", "tau parameter evolution"))
        
        fig.add_trace(go.Scatter(y=q_values, mode='lines', name='q'), row=1, col=1)
        fig.add_trace(go.Scatter(y=tau_values, mode='lines', name='tau'), row=1, col=2)
        
        fig.update_layout(height=600, width=1000, title_text="Parameter Evolution")
        fig.show()

    def plot_wave_function(self, reality_wave_function, q_range, tau_range):
        q_values = np.linspace(q_range[0], q_range[1], 100)
        tau_values = np.linspace(tau_range[0], tau_range[1], 100)
        Q, T = np.meshgrid(q_values, tau_values)
        
        PSI = np.vectorize(reality_wave_function.psi_R)(Q, T)
        
        fig = go.Figure(data=[go.Surface(z=np.abs(PSI), x=Q, y=T)])
        fig.update_layout(title='Reality Wave Function', autosize=False,
                          width=800, height=800,
                          scene=dict(xaxis_title='q', yaxis_title='tau', zaxis_title='|Ψ_R|'))
        fig.show()

    def plot_phase_space(self, data, embedding_dimension=2, delay=1):
        N = len(data)
        embedded_data = np.array([data[i:i+embedding_dimension] 
                                  for i in range(0, N - embedding_dimension + 1, delay)])
        
        fig = go.Figure(data=[go.Scatter3d(x=embedded_data[:, 0],
                                           y=embedded_data[:, 1],
                                           z=embedded_data[:, 2],
                                           mode='markers',
                                           marker=dict(size=2))])
        
        fig.update_layout(title='Phase Space Reconstruction',
                          scene=dict(xaxis_title='X(t)',
                                     yaxis_title='X(t+τ)',
                                     zaxis_title='X(t+2τ)'))
        fig.show()

    def plot_multidimensional_state(self):
        states = self.uo.observation_history[-100:]
        q_values = [state['q'] for state in states]
        tau_values = [state['tau'] for state in states]
        entropy_values = [self.uo.compute_entropy() for _ in states]
        
        fig = go.Figure(data=[go.Scatter3d(x=q_values,
                                           y=tau_values,
                                           z=entropy_values,
                                           mode='markers+lines',
                                           marker=dict(size=4,
                                                       color=range(len(states)),
                                                       colorscale='Viridis'),
                                           line=dict(color='lightblue', width=2))])
        
        fig.update_layout(title='Multidimensional State Evolution',
                          scene=dict(xaxis_title='q',
                                     yaxis_title='tau',
                                     zaxis_title='Entropy'))
        fig.show()
