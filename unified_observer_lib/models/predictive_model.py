import numpy as np
import tensorflow as tf
from ..core.unified_observer import UnifiedObserver
from ..core.reality_wave_function import RealityWaveFunction
from ..nn.quantum_nn import QuantumNN, quantum_activation

class PredictiveModel:
    def __init__(self, unified_observer, nn_layer_sizes=[64, 32, 1], learning_rate=0.001):
        self.uo = unified_observer
        self.rwf = RealityWaveFunction(unified_observer)
        self.nn = QuantumNN(nn_layer_sizes, activation=quantum_activation)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_history = []
        self.smoothed_loss = None

    def predict(self, input_data, future_tau):
        current_psi = self.rwf.psi_R(self.uo.q, self.uo.tau)
        future_psi = self.rwf.psi_R(self.uo.q, future_tau)
        
        nn_input = np.concatenate([
            input_data,
            [current_psi, future_psi]
        ])
        
        nn_output = self.nn(nn_input)
        
        prediction = nn_output * np.abs(future_psi - current_psi)
        return prediction

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.predict(x, self.uo.tau + 1)
            loss = tf.keras.losses.mean_squared_error(y, predictions)
            loss += sum(self.nn.losses)  # Add regularization losses
        
        grads = tape.gradient(loss, self.nn.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.nn.trainable_variables))
        return loss

    def train(self, X, y, epochs=100, batch_size=32):
        dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size)
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_x, batch_y in dataset:
                batch_loss = self.train_step(batch_x, batch_y)
                epoch_loss += batch_loss
            
            avg_loss = epoch_loss / len(dataset)
            self.loss_history.append(avg_loss)
            
            if self.smoothed_loss is None:
                self.smoothed_loss = avg_loss
            else:
                self.smoothed_loss = 0.9 * self.smoothed_loss + 0.1 * avg_loss
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {self.smoothed_loss}")
            
            # Dynamic parameter adjustment
            self.uo.adapt(self.smoothed_loss)
            self.uo.reflect()

    def evaluate(self, X, y):
        predictions = self.predict(X, self.uo.tau + 1)
        mse = tf.keras.losses.mean_squared_error(y, predictions)
        return mse.numpy()
