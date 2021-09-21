import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.python.keras import activations 

class Autoencoder(Model):
    def __init__(self, architecture):
        super(Autoencoder, self).__init__()
        self.latent_dim = architecture['latent_dim']
        
        # Verify latent_dim
        layer_size, _ = architecture['encoder'][-1]
        if layer_size != self.latent_dim:
            raise Exception('Latent dim does not match with output size of autoencoder encoder')

        # Encoder architecture
        self.encoder = tf.keras.Sequential()
        self.encoder.add(layers.Input(shape=architecture['input_shape']))
        for layer_size, layer_activation in architecture['encoder']:
            self.encoder.add(tf.keras.layers.Dense(layer_size, activation=layer_activation))

        # Decoder architecture
        self.decoder = tf.keras.Sequential()
        for layer_size, layer_activation in architecture['decoder']:
            self.decoder.add(tf.keras.layers.Dense(layer_size, activation=layer_activation))
        self.decoder.add(tf.keras.layers.Dense(architecture['input_shape'], activation='sigmoid'))

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def create_autoencoder(X_train, architecture):
    # Create model
    autoencoder = Autoencoder(architecture)
    # Compile model
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    # Train model
    X_train = tf.constant(X_train)
    autoencoder.fit(X_train, X_train,
                    epochs=architecture['epochs'])
    return autoencoder
