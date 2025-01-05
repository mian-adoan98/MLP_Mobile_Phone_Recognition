# Task 4: Model Selection 
# 
# 3 types of models for building: 
# - model 1: simple CNN model 
# - model 2: CNN model with DeepNet-architecture 
# - model 3: CNN model with GoogleLeNet-architecture # 

# Import dependencies from Tensorflow to create models 
import tensorflow as tf 
from tensorflow import keras 
from keras import models
from keras import layers
import keras_tuner as kt

# - model 1: simple CNN model 
def ConvNeuralNet(input_shape: tuple, brand_names: int, rate:float) -> keras.models.Sequential:
  # Convolutional and pooling layers
  input_layer = layers.Input(input_shape)
  x = layers.Conv2D(32, (3,3), activation="relu")(input_layer)
  x = layers.MaxPool2D((2,2))(x)

  x = layers.Conv2D(64, (3,3), activation="relu")(x)
  x = layers.MaxPool2D((2,2))(x)

  x = layers.Conv2D(32, (3,3), activation="relu")(x)
  x = layers.MaxPool2D((2,2))(x)

  # Fully connected layers 
  x = layers.Flatten()(x)
  x = layers.Dropout(rate)(x)
  x = layers.Dense(128, activation="relu")(x)
  output_layer = layers.Dense(brand_names, activation="softmax")(x)

  # Create model 
  model = models.Model(inputs=input_layer, outputs=output_layer)
  return model 


# - model 2: CNN model with DeepNet-architecture
def DeepNeuralNet(input_shape: tuple, brand_names: int, rate:float) -> keras.models.Sequential:
  # Define input layer
  input_layer = layers.Input(input_shape)

  # Branch 1: 2 convolutional layers + 1 maxpool layers of (2D): with 32
  x1 = layers.Conv2D(32, (3,3), activation="relu")(input_layer)
  x1 = layers.Conv2D(32, (3,3), activation="relu")(x1)
  x1 = layers.MaxPool2D((2,2))(x1)

  # Branch 2: 2 convolutional layers + 1 maxpool layers of (2D): with 64
  x2 = layers.Conv2D(32, (3,3), activation="relu")(x1)
  x2 = layers.Conv2D(32, (3,3), activation="relu")(x2)
  x2 = layers.MaxPool2D((2,2))(x2)

  # Branch 3: 2 convolutional layers + 1 maxpool layers of (2D): with 32
  x3 = layers.Conv2D(32, (3,3), activation="relu")(x2)
  x3 = layers.Conv2D(32, (3,3), activation="relu")(x3)
  x3 = layers.MaxPool2D((2,2))(x3)

  # Fully connected layers: flatten layer, fc dense layer and dropout layer (regularization)
  x = layers.Flatten()(x3)
  x = layers.Dense(128, activation="relu")(x)
  x = layers.Dropout(rate)(x)

  # Output layer
  output_layer = layers.Dense(brand_names, activation="softmax")(x)

  # Implement layers into model
  model = models.Model(inputs=input_layer, outputs=output_layer)
  return model 

# - model 3: CNN model with GoogleLeNet-architecture


# ----------------------------------------- Hyperparameter Tuning ----------------------------------------- #

def DeepNeuralNetHP(input_shape: tuple, brand_names: int, 
                    rate:float, hp) -> keras.models.Sequential:
  # Define input layer and the dropout rate(hyperparameter)
  input_layer = layers.Input(input_shape)
  rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)

  # Branch 1: 2 convolutional layers + 1 maxpool layers of (2D): with 32 (1 layer with hyperparameter)
  x1 = layers.Conv2D(hp.Int('filters_1', min_value=32, max_value=128), (3,3), activation="relu")(input_layer)
  x1 = layers.Conv2D(32, (3,3), activation="relu")(x1)
  x1 = layers.MaxPool2D((2,2))(x1)

  # Branch 2: 2 convolutional layers + 1 maxpool layers of (2D): with 64 (1 layer with hyperparameter)
  x2 = layers.Conv2D(hp.Int('filters_1', min_value=32, max_value=128), (3,3), activation="relu")(x1)
  x2 = layers.Conv2D(32, (3,3), activation="relu")(x2)
  x2 = layers.MaxPool2D((2,2))(x2)

  # Branch 3: 2 convolutional layers + 1 maxpool layers of (2D): with 32 (1 layer with hyperparameter)
  x3 = layers.Conv2D(hp.Int('filters_1', min_value=32, max_value=128), (3,3), activation="relu")(x2)
  x3 = layers.Conv2D(32, (3,3), activation="relu")(x3)
  x3 = layers.MaxPool2D((2,2))(x3)

  # Fully connected layers: flatten layer, fc dense layer and dropout layer (regularization)
  x = layers.Flatten()(x)
  x = layers.Dense(128, activation="relu")(x)
  x = layers.Dropout(rate)(x)

  # Output layer
  output_layer = layers.Dense(brand_names, activation="softmax")(x)

  # Implement layers into model
  model = models.Model(inputs=input_layer, outputs=output_layer)
  return model 
