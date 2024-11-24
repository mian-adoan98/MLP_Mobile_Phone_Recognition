## Model Development Tools 
# 
# function 1: training the model 
# function 2: data augmentation for training and testing sets 
# function 3: visualise model performances #

import pandas as pd
import numpy as np  
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# function 1: data augmentation for traing and testing images 
def data_augmentation(train_path: str,
                      test_path: str,
                      rotation: int,
                      img_size: tuple, 
                      batch_size: int = 32
                      ) -> tuple: 
  
  # define constants: image size, number channels and color code
  # img_size = (224, 224)

  # augment the training data
  train_datagen = ImageDataGenerator(
    rescale=1./255,             # Normalize pixel values to [0, 1]
    rotation_range=rotation,         # Random rotation up to 30 degrees
    width_shift_range=0.2,     # Shift images horizontally by up to 20%
    height_shift_range=0.2,    # Shift images vertically by up to 20%
    shear_range=0.2,           # Shear transformation
    zoom_range=0.2,            # Zoom in or out by up to 20%
    horizontal_flip=True,      # Randomly flip images horizontally
    fill_mode='nearest'        # Fill empty areas created by transformations
  )

  # preprocess the test data
  test_datagen = ImageDataGenerator(rescale=1./255)

  # create generator for training images
  train_generator = train_datagen.flow_from_directory(
    train_path,  # Directory with training images
    target_size=(330, 330), # Resize all images to 224x224
    batch_size=32,
    class_mode='categorical'
  )

  # create generator for testing images 
  test_generator = test_datagen.flow_from_directory(
    test_path,  # Directory with training images
    target_size=(330, 330), # Resize all images to 224x224
    batch_size=32,
    class_mode='categorical'
  )

  print(f"Shape of train images: {train_generator.image_shape}")
  print(f"Number of training samples: {train_generator.samples}")
  print(f"Number of test samples: {test_generator.samples}")
  return train_generator, test_generator


# data augmentation: from source https://medium.com/thedeephub/computer-vision-project-image-classification-with-tensorflow-and-keras-264944d09721 
def augment_data( train_df, valid_df, test_df, batch_size=32):      
  img_size = (256,256)
  channels = 3
  color = 'rgb'

  train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
              rotation_range=30,
              horizontal_flip=True,
              vertical_flip=True,
              brightness_range=[0.5, 1.5])
          
  valid_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
          
  train_generator = train_datagen.flow_from_dataframe(
              train_df,
              x_col='filepaths',
              y_col='labels',
              target_size=img_size,
              color_mode=color,
              batch_size=batch_size,
              shuffle=True,
              class_mode='categorical'
          )
   
  print("Shape of augmented training images:", train_generator.image_shape)
          
  valid_generator = valid_test_datagen.flow_from_dataframe(
              valid_df,
              x_col='filepaths',
              y_col='labels',
              target_size=img_size,
              color_mode=color,
              batch_size=batch_size,
              shuffle=True,
              class_mode='categorical'
          )
         
  print("Shape of validation images:", valid_generator.image_shape)
          
  test_generator = valid_test_datagen.flow_from_dataframe(
              test_df,
              x_col='filepaths',
              y_col='labels',
              target_size=img_size,
              color_mode=color,
              batch_size=batch_size,
              shuffle=False,
              class_mode='categorical'
          )
          
  print("Shape of test images:", test_generator.image_shape)
          
  return train_generator, valid_generator, test_generator