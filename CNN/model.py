import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = keras.layers.Conv2D(16, 3, padding='same', activation='relu')
        self.bat1 = keras.layers.BatchNormalization()
        self.pool1 = keras.layers.MaxPooling2D()
        self.conv2 = keras.layers.Conv2D(32, 3, padding='same', activation='relu')
        self.bat2 = keras.layers.BatchNormalization()
        self.pool2 = keras.layers.MaxPooling2D()
        self.conv3 = keras.layers.Conv2D(64, 3, padding='same', activation='relu')
        self.bat3 = keras.layers.BatchNormalization()
        self.pool3 = keras.layers.MaxPooling2D()
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(128, activation='relu')
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
    
#####################################################################################################################    
    
class Vgg_Block1(tf.keras.layers.Layer):
    def __init__(self, n_filter, filter_size=(3, 3), reduce_size=True):
        super(Vgg_Block1, self).__init__()
        self.vgg_conv1 = tf.keras.layers.Conv2D(n_filter, filter_size, padding='same', activation='relu')
        self.vgg_conv2 = tf.keras.layers.Conv2D(n_filter, filter_size, padding='same', activation='relu')
        self.vgg_pool = tf.keras.layers.MaxPool2D((2, 2))
        
    def call(self, inputs):
        x = self.vgg_conv1(inputs)
        x = self.vgg_conv2(x)
        x = self.vgg_pool(x)
        return x
    
class Vgg_Block2(tf.keras.layers.Layer):
    def __init__(self, n_filter, filter_size=(3, 3), reduce_size=True):
        super(Vgg_Block2, self).__init__()
        self.vgg_conv1 = tf.keras.layers.Conv2D(n_filter, filter_size, padding='same', activation='relu')
        self.vgg_conv2 = tf.keras.layers.Conv2D(n_filter, filter_size, padding='same', activation='relu')
        self.vgg_conv3 = tf.keras.layers.Conv2D(n_filter, filter_size, padding='same', activation='relu')
        self.vgg_pool = tf.keras.layers.MaxPool2D((2, 2))
        
    def call(self, inputs):
        x = self.vgg_conv1(inputs)
        x = self.vgg_conv2(x)
        x = self.vgg_conv3(x)
        x = self.vgg_pool(x)
        return x
"""        
        else n_conv == 3:
            self.vgg_conv1 = tf.keras.layers.Conv2D(n_filter, filter_size, padding='same', activation='relu')
            self.vgg_conv2 = tf.keras.layers.Conv2D(n_filter, filter_size, padding='same', activation='relu')
            self.vgg_conv3 = tf.keras.layers.Conv2D(n_filter, filter_size, padding='same', activation='relu')
            self.vgg_pool = tf.keras.layers.MaxPool2D((2, 2))
        
        def call(self, inputs):
            x = self.vgg_conv1(inputs)
            x = self.vgg_conv2(x)
            x = self.vgg_conv3(x)
            x = self.vgg_pool(x)
            return x
"""    
    #밑에서 연결을 시켜야하는데 for, if를 쓰면서 어떻게 연결을 시키지
    #subclass에서 반복문을 쓸 수 있나
# VGG set
    #11 >> 64 | 128 | 256 256 | 512 512 | 512 512 |
    #13 >> 64 64 | 128 128 | 256 256 | 512 512 | 512 512 |
    #16 >> 64 64 | 128 128 | 256 256 256 | 512 512 512 | 512 512 512 |
    #19 >> 64 64 | 128 128 | 256 256 256 256 | 512 512 512 512 | 512 512 512 512 |
    # 28 28
class VggModel(tf.keras.Model):
    def __init__(self):
        super(VggModel, self).__init__() 
        self.vgg_block1 = Vgg_Block1(32)
        self.vgg_block2 = Vgg_Block1(64)
        self.vgg_block3 = Vgg_Block2(128)
        self.vgg_block4 = Vgg_Block2(128)
        #self.vgg_block5 = Block2(256)
        self.flatten = tf.keras.layers.Flatten()
        #self.dense1 = tf.keras.layers.Dense(64, activation = 'relu')
        self.dense1 = tf.keras.layers.Dense(64, activation = 'relu')
        #self.drop1 = keras.layers.Dropout(rate=0.1)
        self.dense2 = tf.keras.layers.Dense(5)
    
    """
    def bulid(self, inputs):
        self.vgg_block1 = vgg_block(inputs, 2, 64)
        self.vgg_block2 = vgg_block(self.vgg_block1, 2, 128)
        self.vgg_block3 = vgg_block(self.vgg_block2, 3, 256)
        self.vgg_block4 = vgg_block(self.vgg_block3, 3, 512)
    """
    def call(self, inputs):
        vnet = self.vgg_block1(inputs)
        vnet = self.vgg_block2(vnet)
        vnet = self.vgg_block3(vnet)
        vnet = self.vgg_block4(vnet)
        #vnet = self.vgg_block5(vnet)
        vnet = self.flatten(vnet)
        vnet = self.dense1(vnet)
        #vnet = self.drop1(vnet)
        vnet = self.dense2(vnet)
        return vnet
    
    #def call(self, inputs):
        #vnet = self.vgg_block1(inputs)
        #vnet = self.vgg_block1(vnet)
        #vnet = self.vgg_block1(vnet)
        #vnet = self.vgg_block2(vnet)
        #vnet = self.vgg_block3(vnet)
        #vnet = self.vgg_block4(vnet)
        #vnet = self.flatten(vnet)
        #vnet = self.dense1(vnet)
        #vnet = self.dense2(vnet)
        #return vnet
    """
    def vgg_block(self, in_layer, n_conv, n_filter, filter_size=(3, 3), reduce_size=True):
        layer = in_layer
        for i in range(n_conv):
            layer = tf.keras.layers.Conv2D(n_filter, filter_size, padding='SAME', activation='relu')(layer)
                
        if reduce_size:
            layer = tf.keras.layers.MaxPool2D((2,2))(layer)
        return layer
    """
    
##########################################################################################################################

class ConvModule(tf.keras.layers.Layer):
    def __init__(self, n_filter, filter_size, strides, padding='same'):
        super(ConvModule, self).__init__()
        
        self.conv = tf.keras.layers.Conv2D(n_filter, filter_size, strides = strides, padding = padding)
        self.bn = tf.keras.layers.BatchNormalization()
        
    def call(self, inputs, training = False):
        
        x = self.conv(inputs)
        x = self.bn(x, training = training)
        x = tf.nn.relu(x)
        
        return x
# 32 32 3
class InceptionModule(tf.keras.layers.Layer):
    def __init__(self, conv_11_size, conv_33_size):
        super(InceptionModule, self).__init__()
        
        self.conv1 = ConvModule(conv_11_size, filter_size=(1,1), strides=(1,1))
        self.conv2 = ConvModule(conv_33_size, filter_size=(3,3), strides=(1,1))
        self.cat = tf.keras.layers.Concatenate()
        
    def call(self, inputs, training = False):
        x_11 = self.conv1(inputs)
        x_33 = self.conv2(inputs)
        x = self.cat([x_11, x_33])
        return x
    
class DownsampleModule(tf.keras.layers.Layer):
    def __init__(self, filter_size):
        super(DownsampleModule, self).__init__()
        
        self.conv3 = ConvModule(filter_size, filter_size=(3,3), strides=(2,2), padding='valid')
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2))
        self.cat = tf.keras.layers.Concatenate()
        
    def call(self, inputs, training = False):
        conv_x = self.conv3(inputs, training=training)
        pool_x = self.pool(inputs)
        
        return self.cat([conv_x, pool_x])
    
class MiniInception(tf.keras.Model):
    def __init__(self):
        super(MiniInception, self).__init__()
        
        self.conv_block = ConvModule(96, (3,3), (1,1))
        
        self.inception_block1 = InceptionModule(32, 32)
        self.inception_block2 = InceptionModule(32, 48)
        self.downsample_block1 = DownsampleModule(80)
        
        self.inception_block3 = InceptionModule(112, 48)
        self.inception_block4 = InceptionModule(96, 64)
        self.inception_block5 = InceptionModule(80, 80)
        self.inception_block6 = InceptionModule(48, 96)
        self.downsample_block2 = DownsampleModule(96)
        
        self.inception_block7 = InceptionModule(176, 160)
        self.inception_block8 = InceptionModule(176, 160)
        
        self.avg_pool = tf.keras.layers.AveragePooling2D((7,7))
        
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(5, activation='softmax')
        
    def call(self, inputs, training=False, **kwargs):
        
        x = self.conv_block(inputs)
        x = self.inception_block1(x)
        x = self.inception_block2(x)
        x = self.downsample_block1(x)
        
        x = self.inception_block3(x)
        x = self.inception_block4(x)
        x = self.inception_block5(x)
        x = self.inception_block6(x)
        x = self.downsample_block2(x)
        
        x = self.inception_block7(x)
        x = self.inception_block8(x)
        
        x = self.avg_pool(x)
        
        x = self.flatten(x)
        x = self.dense(x)
        
        return x

#################################################################################################################################

#stride
class ResBlock(tf.keras.layers.Layer):
    def __init__(self, n_filter, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(n_filter, kernel_size = (3, 3), strides=stride, padding='same')
        #print(self.conv1.strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(n_filter, kernel_size = (3, 3), strides = 1, padding = 'same')
        #print(self.conv2.strides)
        self.bn2 = tf.keras.layers.BatchNormalization()
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(n_filter,
                                                       kernel_size = (1, 1),
                                                       strides = stride))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)
        
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        output = tf.nn.relu(tf.keras.layers.add([residual, x]))
        
        return output
    
class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, n_filter, stride=1):
        super(BottleNeck, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(filters=n_filter, kernel_size = (1, 1), strides = 1, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=n_filter, kernel_size = (3, 3), strides = stride, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters=n_filter*4, kernel_size = (1, 1), strides = 1, padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()
        
        self.downsample = tf.keras.Sequential()
        self.downsample.add(tf.keras.layers.Conv2D(filters=n_filter*4, kernel_size = (1, 1), strides = stride))
        self.downsample.add(tf.keras.layers.BatchNormalization())
        
    def call(self, inputs):
        residual = self.downsample(inputs)
        
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        output = tf.nn.relu(tf.keras.layers.add([residual, x]))
        
        return output
    
def make_basic_block_layer(n_filter, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(ResBlock(n_filter, stride=stride))
    
    for _ in range(1, blocks):
        res_block.add(ResBlock(n_filter, stride=1))
    
    return res_block

def make_bottleneck_layer(n_filter, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BottleNeck(n_filter, stride = stride))
    
    for _ in range(1, blocks):
        res_block.add(BottleNeck(n_filter, stride=1))
        
    return res_block
    
#set    
    #basic block
    # 18 2 2 2 2
    # 34 3 4 6 3
    #bottleneck
    # 50 3 4 6 3
    # 101 3 4 23 3
    # 152 3 8 36 3
    # 224 224
    
class ResNet(tf.keras.Model):
    def __init__(self):
        super(ResNet, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(64, (7, 7), strides=2, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D((3, 3), 2, padding='same')
        
        self.layer1 = make_bottleneck_layer(64, blocks=3)
        self.layer2 = make_bottleneck_layer(128, blocks=4, stride=2)
        self.layer3 = make_bottleneck_layer(256, blocks=6, stride=2)
        self.layer4 = make_bottleneck_layer(512, blocks=3, stride=2)
        
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(5, activation='softmax')
    
    def call(self, inputs, training=False):
        
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.dense1(x)
        
        return x
"""    
if __name__ == "__main__":
    
    
    ResBlock(1)
    
"""