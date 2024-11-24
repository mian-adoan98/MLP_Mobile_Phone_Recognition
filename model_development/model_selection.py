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
from keras.layers import GlobalAveragePooling2D

# model 1: simple convolutional neural network 
class CNNModel(Model):
  def __init__(self, 
               filter_size: tuple, 
               window_size: tuple,
               num_classes: int,
               dropout_rate: int = 0.4) -> object:
    super().__init__()
    # define attributes for layers 
    self.filter_size = filter_size
    self.window_size = window_size
    self.num_classes = num_classes

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
    # self.glober_avg_pooling_layer = GlobalAveragePooling2D()
    self.flatten = Flatten()

    # output layers + fully connected layers
    self.fc1 = Dense(128, activation="relu")
    self.dropout = Dropout(dropout_rate)
    self.output_layer = Dense(num_classes, activation="softmax")

  def call(self, inputs):  
    # forward pass: applying each layer in sequence
    x = self.conv1(inputs)
    x = self.maxpool1(x)
    print(f"Shape after 1st block: {x.shape}")  # Debug
    
    x = self.conv2(x)
    x = self.maxpool2(x)
    print(f"Shape after 2nd block: {x.shape}")  # Debug
    
    x = self.conv3(x)
    x = self.maxpool3(x)
    print(f"Shape after 3rd block: {x.shape}")  # Debug

    # apply flatten layer
    x = self.flatten(x)
    print(f"Shape after flatten: {x.shape}")  # Debug

    # Apply dense and dropout layers
    x = self.fc1(x)  # Fully connected layer
    print(f"Shape after FC1: {x.shape}")  # Debug
    
    x = self.dropout(x)
    outputs = self.output_layer(x)
    print(f"Shape after output layer: {outputs.shape}")  # Debug
    
    return outputs  
  
  # Method to return the config of the model
  def get_config(self):
      config = super(CNNModel, self).get_config()
      config.update({"num_classes": self.output_layer.units})  # Add custom parameters
      return config

  # Method to load the model using config
  @classmethod
  def from_config(cls, config):
      num_classes = config.pop('num_classes')
      return cls(num_classes=num_classes)

# model 2: model with VGG-architecture 
class VGGModel(Model):
  def __init__(self, 
               filter_size: tuple, 
               window_size: tuple,
               num_classes: int,
               dropout_rate: int = 0.4) -> object:
    # define attributes for layers 
    self.filter_size = filter_size
    self.window_size = window_size
    self.num_classes = num_classes

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