import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        #self.conv1 = keras.layers.Conv2D(16, 3, padding='same', activation='relu')
        self.conv1 = keras.layers.Conv2D(32, 3, padding='same', activation='relu')
        self.bat1 = keras.layers.BatchNormalization()
        self.pool1 = keras.layers.MaxPooling2D()
        #self.conv2 = keras.layers.Conv2D(32, 3, padding='same', activation='relu')
        self.conv2 = keras.layers.Conv2D(64, 3, padding='same', activation='relu')
        self.bat2 = keras.layers.BatchNormalization()
        self.pool2 = keras.layers.MaxPooling2D()
        #self.conv3 = keras.layers.Conv2D(64, 3, padding='same', activation='relu') 
        self.conv3 = keras.layers.Conv2D(128, 3, padding='same', activation='relu')
        self.bat3 = keras.layers.BatchNormalization()
        self.pool3 = keras.layers.MaxPooling2D()
        self.flatten = keras.layers.Flatten()
        #self.dense1 = keras.layers.Dense(128, activation='relu')
        self.dense1 = keras.layers.Dense(256, activation='relu') 
        #self.drop1 = keras.layers.Dropout(rate=0.2)
        self.dense2 = keras.layers.Dense(5)
    
    def call(self, inputs, training=False):
        net = self.conv1(inputs)
        net = self.bat1(net)
        net = tf.nn.relu(net)
        net = self.pool1(net)
        net = self.conv2(net)
        net = self.bat2(net)
        net = tf.nn.relu(net)
        net = self.pool2(net)
        net = self.conv3(net)
        net = self.bat3(net)
        net = tf.nn.relu(net)
        net = self.pool3(net)
        net = self.flatten(net)
        net = self.dense1(net)
        #net = self.drop1(net)
        net = self.dense2(net)
        return net
