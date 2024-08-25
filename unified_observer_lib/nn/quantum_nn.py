import tensorflow as tf
import numpy as np

class QuantumInspiredLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, use_phase=False):
        super(QuantumInspiredLayer, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_phase = use_phase

    def build(self, input_shape):
        self.w_real = self.add_weight(
            name='w_real',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True)
        self.w_imag = self.add_weight(
            name='w_imag',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True)
        self.b_real = self.add_weight(
            name='b_real',
            shape=(self.units,),
            initializer='zeros',
            trainable=True)
        self.b_imag = self.add_weight(
            name='b_imag',
            shape=(self.units,),
            initializer='zeros',
            trainable=True)
        if self.use_phase:
            self.phase = self.add_weight(
                name='phase',
                shape=(self.units,),
                initializer='random_uniform',
                trainable=True)

    def call(self, inputs):
        inputs_real = tf.math.real(inputs)
        inputs_imag = tf.math.imag(inputs)
        
        outputs_real = tf.matmul(inputs_real, self.w_real) - tf.matmul(inputs_imag, self.w_imag) + self.b_real
        outputs_imag = tf.matmul(inputs_real, self.w_imag) + tf.matmul(inputs_imag, self.w_real) + self.b_imag
        
        outputs = tf.complex(outputs_real, outputs_imag)
        
        if self.use_phase:
            outputs *= tf.exp(1j * self.phase)
        
        if self.activation is not None:
            outputs = self.activation(outputs)
        
        return outputs

class QuantumNN(tf.keras.Model):
    def __init__(self, layer_sizes, activation='tanh', use_phase=False, dropout_rate=0.2):
        super(QuantumNN, self).__init__()
        self.layers_list = []
        for units in layer_sizes[:-1]:
            self.layers_list.append(QuantumInspiredLayer(units, activation, use_phase))
            self.layers_list.append(tf.keras.layers.Dropout(dropout_rate))
            self.layers_list.append(tf.keras.layers.BatchNormalization())
        self.layers_list.append(QuantumInspiredLayer(layer_sizes[-1], use_phase=use_phase))

    def call(self, inputs):
        x = tf.cast(inputs, dtype=tf.complex64)
        for layer in self.layers_list:
            x = layer(x)
        return tf.abs(x)

def quantum_activation(x, phase_shift=0.1):
    magnitude = tf.abs(x)
    phase = tf.angle(x) + phase_shift
    return tf.complex(magnitude * tf.cos(phase), magnitude * tf.sin(phase))
