# Task 4: Model Selection 
from tensorflow import keras 
from keras.models import Sequential 
from keras.models import Model

# Building 3 models for assignment: 
# - simple convolutional neural network 
# - convolutional neural network based on its chose architecture: VGG, ResNet 
# #

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dense, BatchNormalization, Add, Input, Dropout

# model 1: simple convolutional neural network 
class CNNModel(Model):
  def __init__(self, 
               filter_size: tuple, 
               window_size: tuple,
               num_classes: int,
               n_images) -> object:
    # define attributes for layers 
    self.filter_size = filter_size
    self.window_size = window_size
    self.num_classes = num_classes
    self.n_images = n_images 

    # branch 1: convolutional: 2 layers; maxpooling: 1 layer
    self.conv1 = Conv2D(32, self.filter_size, activation="relu", padding="same")
    self.maxpool1 = MaxPooling2D(self.window_size)

    # branch 2: convolutional: 2 layers; maxpooling: 1 layer
    self.conv2 = Conv2D(64, self.filter_size, activation="relu", padding="same")
    self.maxpool2 = MaxPooling2D(self.window_size)

    # branch 3: convolutional: 2 layers; maxpooling: 1 layer
    self.conv3 = Conv2D(128, self.filter_size, activation="relu", padding="same")
    self.maxpool3 = MaxPooling2D(self.window_size)

    # reduce dimensionality to 1D
    self.flatten = Flatten()

    # output layers + fully connected layers
    self.fc1 = Dense(128, activation="relu")
    self.dropout = Dropout(0.4)
    self.output_layer = Dense(num_classes, activation="softmax")

  def call(self, n_images):
    # define inputs
    inputs = Input(shape=(n_images,))
    
    # forward pass: applying each layer in sequence
    x = self.conv1(inputs)
    x = self.maxpool1(x)
    x = self.conv2(inputs)
    x = self.maxpool2(x)
    x = self.conv3(inputs)
    x = self.maxpool3(x)
    x = self.flatten(x)
    x = self.dropout(x)
    x = self.output_layer(x)

    return self.output_layer(x)

class VGGModel(Model):
  def __init__(self, 
               filter_size: tuple, 
               window_size: tuple,
               num_classes: int,
               n_images) -> object:
    # define attributes for layers 
    self.filter_size = filter_size
    self.window_size = window_size
    self.num_classes = num_classes
    self.n_images = n_images 

    # branch 1: convolutional: 2 layers; maxpooling: 1 layer
    self.conv11 = Conv2D(32, self.filter_size, activation="relu", padding="same")
    self.conv21 = Conv2D(32, self.filter_size, activation="relu", padding="same")
    self.maxpool1 = MaxPooling2D(self.window_size)

    # branch 1: convolutional: 2 layers; maxpooling: 1 layer
    self.conv21 = Conv2D(64, self.filter_size, activation="relu", padding="same")
    self.conv22 = Conv2D(64, self.filter_size, activation="relu", padding="same")
    self.maxpool2 = MaxPooling2D(self.window_size)

    # branch 1: convolutional: 2 layers; maxpooling: 1 layer
    self.conv31 = Conv2D(128, self.filter_size, activation="relu", padding="same")
    self.conv32 = Conv2D(128, self.filter_size, activation="relu", padding="same")
    self.maxpool3 = MaxPooling2D(self.window_size)

    # reduce dimensionality to 1D
    self.flatten = Flatten()

    # output layers + fully connected layers
    self.fc1 = Dense(128, activation="relu")
    self.dropout = Dropout(0.4)
    self.output_layer = Dense(num_classes, activation="softmax")

  def call(self, n_images):
    # define inputs
    inputs = Input(shape=(n_images,))
    
    # forward pass 1: applying convolutional and pooling layer in sequence
    x = self.conv1(inputs)
    x = self.maxpool1(x)
    x = self.conv2(x)
    x = self.maxpool2(x)
    x = self.conv3(x)
    x = self.maxpool3(x)

    # forward pass 2: applying flatten, dropout and output_layer in sequence
    x = self.flatten(x)
    x = self.dropout(x)
    x = self.output_layer(x)

    return self.output_layer(x)