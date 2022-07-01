from ...config.config import model_config

import tensorflow as tf
from tensorflow.keras.layers import Layer, LSTM, Dense


class DFRNNNoise(Layer):
    def __init__(self, config=model_config, **kwargs):
        super(DFRNNNoise, self).__init__(**kwargs)
        self.config = config
        self.lstm = LSTM(
            units=config.noise_lstm_units, 
            return_sequences=True, 
            return_state=True, 
            activation='relu'
            )
        self.dense = Dense(units=1)

    def call(self, inputs):
        hidden_state, _, _ = self.lstm(inputs)
        _sigma = self.dense(hidden_state)
        sigma = tf.squeeze(_sigma)

        return sigma
