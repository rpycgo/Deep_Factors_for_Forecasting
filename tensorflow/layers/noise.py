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
        _, _hidden_state, cell_state = self.lstm(inputs)
        hidden_state = self.dense(_hidden_state)
        _sigma = self.dense(hidden_state)
        sigma = tf.reshape(_sigma, shape=(self.config.batch_size, -1))

        return sigma
