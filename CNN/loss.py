import tensorflow as tf

from tensorflow import keras
#from tensorflow.keras import layers


def loss_fn(label, logits):
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = loss(label, logits)  
    return loss
