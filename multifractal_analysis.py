import numpy as np
from scipy import stats

class MultifractalAnalyzer:
    def __init__(self, q_values=None):
        self.q_values = q_values if q_values is not None else np.linspace(-5, 5, 101)

    def compute_fluctuation_function(self, data, q, scale):
        segments = len(data) // scale
        F_q = 0
        for i in range(segments):
            segment = data[i*scale:(i+1)*scale]
            F_q += np.power(np.sum(np.abs(segment - np.mean(segment))**2), q/2)
        return np.power(F_q / segments, 1/q)

    def multifractal_detrended_fluctuation_analysis(self, data, scales):
        F_q_s = np.zeros((len(self.q_values), len(scales)))
        for i, q in enumerate(self.q_values):
            for j, scale in enumerate(scales):
                F_q_s[i, j] = self.compute_fluctuation_function(data, q, scale)
        return F_q_s

    def compute_hurst_exponent(self, F_q_s, scales):
        H_q = np.zeros(len(self.q_values))
        for i, _ in enumerate(self.q_values):
            slope, _, _, _, _ = stats.linregress(np.log(scales), np.log(F_q_s[i, :]))
            H_q[i] = slope
        return H_q

    def compute_multifractal_spectrum(self, H_q):
        tau_q = self.q_values * H_q - 1
        alpha = np.diff(tau_q) / np.diff(self.q_values)
        f_alpha = self.q_values[:-1] * alpha - tau_q[:-1]
        return alpha, f_alpha

    def analyze(self, data, scales):
        F_q_s = self.multifractal_detrended_fluctuation_analysis(data, scales)
        H_q = self.compute_hurst_exponent(F_q_s, scales)
        alpha, f_alpha = self.compute_multifractal_spectrum(H_q)
        return {
            'H_q': H_q,
            'alpha': alpha,
            'f_alpha': f_alpha
        }

    def interpret_results(self, results):
        H_q = results['H_q']
        alpha = results['alpha']
        f_alpha = results['f_alpha']
        
        interpretation = "Multifractal Analysis Results:\n"
        
        # Interpret Hurst exponents
        if np.max(H_q) - np.min(H_q) > 0.2:
            interpretation += "- The time series exhibits multifractal behavior.\n"
        else:
            interpretation += "- The time series shows monofractal characteristics.\n"
        
        # Interpret multifractal spectrum
        width = np.max(alpha) - np.min(alpha)
        interpretation += f"- The width of the multifractal spectrum is {width:.2f}, "
        if width > 0.5:
            interpretation += "indicating a rich, complex structure with multiple scaling behaviors.\n"
        else:
            interpretation += "suggesting a relatively simple scaling structure.\n"
        
        # Interpret f(alpha) curve
        max_f_alpha = np.max(f_alpha)
        interpretation += f"- The maximum of f(alpha) is {max_f_alpha:.2f}, "
        if max_f_alpha > 0.9:
            interpretation += "indicating a high degree of multifractality.\n"
        else:
            interpretation += "suggesting a moderate to low degree of multifractality.\n"
        
        return interpretation
