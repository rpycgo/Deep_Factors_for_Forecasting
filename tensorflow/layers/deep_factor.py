from ...config.config import model_config

import tensorflow as tf
from tensorflow.keras.layers import Layer, LSTM, Dense


class DeepFactor(Layer):
    def __init__(self, config=model_config, **kwargs):
        super(DeepFactor, self).__init__(name='deep_factor')
        self.config = config
        self.global_factors = [
            LSTM(
                units=config.deep_factor_lstm_units, 
                return_sequences=True,
                activation='relu'
                ) for _ in range(self.config.k)
            ]
        self.dense = Dense(units=1)

    def call(self, inputs):
        _output = [global_factor(inputs) for global_factor in self.global_factors]
        _output = [self.dense(__output) for __output in _output]
        _output = tf.stack(_output)
        _output = tf.transpose(_output, perm=(1, 2, 3, 0))
        _output = tf.squeeze(_output)
        
        return tf.math.reduce_mean(_output, axis=-1, keepdims=True)
