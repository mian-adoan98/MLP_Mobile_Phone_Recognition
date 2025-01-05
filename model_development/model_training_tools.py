## Model Development Tools 
# 
# Branch 1: Model Training
# function 1: training the model 
# function 2: data augmentation for training and testing sets 
# function 3: visualise model performances 
# 
# Brand 2: Model Evaluation
# function 1: visualise model's perforamnce #

import os 
import pandas as pd
import numpy as np  
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# ------------------------------------------------------- Model Training ----------------------------------------------

# function 1: data augmentation for traing and testing images 
def data_preprocessing(dataframe: pd.DataFrame,
                      train_size: int,
                      rotation: int,
                      img_size: tuple, 
                      batch_size: int = 32
                      ) -> tuple: 
  
  # define constants: image size, number channels and color code
  # Split dataset into training and testing set
  train_df = dataframe.sample(frac=train_size, random_state=52)
  test_df = dataframe.drop(train_df.index)

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
  train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,  # Dataframe with training images
    x_col="Image_paths",
    y_col="Class_names",
    target_size=img_size, # Resize all images to 224x224
    batch_size=batch_size,
    class_mode='categorical'
  )

  # create generator for testing images 
  test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df, # Dataframe with testing images
    x_col="Image_paths", 
    y_col="Class_names",   
    target_size=img_size, # Resize all images to 224x224
    batch_size=batch_size,
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

# implement function: resizing images (NOT UP-TO-DATE)
def resize_images():

  # Paths to your input and output directories
  input_folder = "path/to/input_images"
  output_folder = "path/to/output_images"

  # Ensure the output folder exists
  os.makedirs(output_folder, exist_ok=True)

  # Desired size
  new_size = (228, 228)

  # Resize images
  for index, file_name in enumerate(os.listdir(input_folder)):
    if file_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):  # Check for valid image files
      input_path = os.path.join(input_folder, file_name)
      output_path = os.path.join(output_folder, file_name)

      # Open, resize, and save the image
      with Image.open(input_path) as img:
          resized_img = img.resize(new_size, Image.ANTIALIAS)  # Use ANTIALIAS for better quality
          resized_img.save(output_path)

      print(f"Resized image {index + 1}: {file_name} -> {output_path}")
      if index + 1 == 10:  # Process only the first 10 images
          break

# implement function: training the model
def model_training(training_set, 
                   testing_set,
                   batch_size, 
                   epochs, index, model) :
  # Check if the training is suitable for given epoch
  if epochs > 5 and epochs <= 10: 
    # Train the model 
    model_hist = model.fit(
      training_set,
      epochs=epochs,
      batch_size=batch_size,
      validation_data=testing_set,
    )
  elif epochs > 10 and epochs < 20:
    # Train the model 
    model_hist = model.fit(
      training_set,
      epochs=epochs,
      batch_size=batch_size,
      validation_data=testing_set,
    )
  else:
    print(f"Epoch {epochs} is out of range --> choose a new epoch")
  
  # Save performance metrics into dataset
  allocated_dir = "model_development/performance_data"
  if not os.path.exists(allocated_dir): 
    os.makedirs(allocated_dir)

  perform_metrics = model_hist.history
  perfm_df = pd.DataFrame()
  perfm_df.to_csv(f"performance_data{index}.csv")





# ---------------------------------------------------------- Model Evaluation --------------------------------------