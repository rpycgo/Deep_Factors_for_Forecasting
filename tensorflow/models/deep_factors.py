from ...config.config import model_config
from ..layers.noise import DFRNNNoise
from ..layers.deep_factor import DeepFactor

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.models import Model


class DFRNN(Model):
    def __init__(self, config=model_config, **kwargs):
        super(DFRNN, self).__init__(name='DFRNN', **kwargs)
        self.config = model_config
        self.noise = DFRNNNoise(name='noise')
        self.global_factor = DeepFactor(name='global_factor')
        self.embedding = Dense(units=config.k, name='embedding')

    def call(self, inputs):
        _mu = self.global_factor(inputs) # batch_size, time_seq, 1
        _mu = self.embedding(_mu)   # batch_size, time_seq, k
        mu = tf.math.reduce_mean(_mu, axis=-1)   # batch_size, time_seq
        sigma = self.noise(inputs)  # batch_size, time_seq

        return mu, sigma

    def sample(self, inputs, num_samples=100):
        mu, sigma = self(inputs)
        _sampled_data = tf.stack([tf.random.normal(shape=inputs.shape[:2], mean=mu, stddev=sigma) for _ in range(num_samples)])
        sampled_data = tf.math.reduce_mean(_sampled_data, axis=0)

        return sampled_data

