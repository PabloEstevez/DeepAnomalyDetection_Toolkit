import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers

class AutoEncoder(Model):
  def __init__(self, hidden_activation, output_activation, hidden_size, bottleneck_size, input_size, sparsity):
    super(AutoEncoder, self).__init__()
    self.encoder = tf.keras.Sequential([
      Dense(hidden_size, activation=hidden_activation),
      # Dense(64, activation=hidden_activation),
      Dense(bottleneck_size, activation=hidden_activation, activity_regularizer=regularizers.l1(sparsity))]) # 

    self.decoder = tf.keras.Sequential([
      # Dense(64, activation=hidden_activation),
      Dense(hidden_size, activation=hidden_activation),
      Dense(input_size, activation=output_activation)])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded