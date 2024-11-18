# Image processing: Function Declaration script

import os 
import glob
import random 
import pandas as pd 
import numpy as np 
import requests
import shutil
from urllib.parse import urlparse

# implement function 0 --> load collected files for processing
def data_loading(directory: str, content_type: str) -> list:
  # define list for storing content
  content_list = []

  # open the selected file
  with open(directory, "r") as infile:
    for line in infile:
      content_list.append(line)
    
  # convert content list into numpy-array for quick processing
  content = np.array(content_list, dtype="object")
  return content

# implement function 1 --> construct a column list
def construct_feature(feature_name: str, sample: list) -> dict: 
  feature_df = pd.DataFrame()
  feature_df[feature_name] = pd.Series(sample)
  return feature_df

# implement function 2 --> renaming files from selected directory 
def rename_images(directory: str, files: list):
  # Filter only image files
  image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
  image_files = [f for f in files if any(f.endswith(ext) for ext in image_extensions)]

  # Sort the image files numerically (this will sort by the number in the filename)
  # Extract the numeric part of the filename and sort based on that
  sorted_files = sorted(image_files, key=lambda f: int(f.split('_')[1].split('.')[0]))

  # Rename the files with zero-padding to ensure correct lexicographical order
  for idx, file in enumerate(sorted_files):
      # Construct the new file name with zero-padding (adjust padding if needed)
      new_name = f"image_{idx:03d}.jpg"  # This will ensure names like image_001.jpg, image_002.jpg, etc.

      # Get the full file path
      old_path = os.path.join(directory, file)
      new_path = os.path.join(directory, new_name)

      # Rename the file
      os.rename(old_path, new_path)

      print(f"Renamed '{file}' to '{new_name}'")

# # implement function 3 --> move images to other (version 2) (CONTINUE WORKING)(!!)
# def distribute_images(source_folder: str, 
#                       destination_folder: str, 
#                       df_file: str):
  
#   # extract class folders into list: list
#   # ensure the destination folder exists 
#   if not os.path.exists(destination_folder): 
#     os.makedirs(destination_folder)  

#   # load the dataset
#   class_folders = os.listdir(source_folder)
#   image_df = pd.read_csv(df_file, index_col=0)
#   image_label = image_df[["Image_file", "Company"]].values

#   # iterate
#   for image_name, label in image_label:
#     # create class folder path
#     source_path = os.path.join(source_folder, image_name)
#     class_folder = os.path.join(destination_folder, label)

#     # ensure the class folders exist
#     if not os.path.exists(class_folder):
#       os.makedirs(class_folder)
    
#     # move images to the respective label folder
#     if os.path.exists(class_folder):
#       shutil.move(source_path, os.path.join(class_folder, image_name))
#       print(f"Moved {image_name} to {class_folder}")
#     else:
#       print(f"File {image_name} does not exist in {class_folder}")
    
#   # implement iteration: rearrange and distribute the images based on
#   # -> create the path name by joining image name
#   # -> #

# implement function 3 (VERSION 2)
def distribute_images(source_folder: str, 
                      destination_folder: str, 
                      df_file: str,
                      update_train_size: int,
                      update_test_size: int):
  """
  Distribute images into train, test, and validation folders based on their labels.

  Parameters:
  - source_folder (str): Path to the folder containing images.
  - destination_folder (str): Path to the folder where images will be distributed.
  - df_file (str): CSV file containing 'Image_file' and 'Company' columns.
  """
  # Ensure the destination folder exists
  if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

  # Load the dataset
  image_df = pd.read_csv(df_file)
  image_labels = image_df[["Image_file", "Company"]].values  # Get image-label pairs

  # Shuffle the dataset for randomness
  random.shuffle(image_labels)

  # Split ratios
  train_ratio = update_train_size
  test_ratio = update_test_size  # Optionally include validation from test later
  total_images = len(image_labels)

  train_size = int(total_images * train_ratio)
  test_size = int(total_images * test_ratio)
  # test_size = total_images - train_size

  # Split the dataset into train and test
  train_data = image_labels[:train_size]
  test_data = image_labels[train_size:]

  # Prepare folders for train, test, and optionally validation
  splits = {"train": train_data, "test": test_data}
  for split in splits:
    for _, label in splits[split]:  
      class_folder = os.path.join(destination_folder, split, label)
      if not os.path.exists(class_folder):
        os.makedirs(class_folder)

  # Move images to respective folders
  for split, data in splits.items():
    for image_name, label in data:
      source_path = os.path.join(source_folder, image_name)
      dest_path = os.path.join(destination_folder, split, label, image_name)

      if os.path.exists(source_path):  # Check if image exists
        shutil.copy(source_path, dest_path)
        print(f"Moved {image_name} to {os.path.join(split, label)}")
      else:
        print(f"Image {image_name} not found in source folder.")

# implement function 4: remove images from current folder 
def remove_images(current_folder: str):
  # Clear the distributed folder
  if os.path.exists(current_folder):
      shutil.rmtree(current_folder) 

# implement function 5: determine number of images in a folder 
def det_folder_size(sel_path: str) -> pd.DataFrame: 
  # dictionary to store number of images per class folder 
  image_counter = {}

  # loop through each class folder in the main directory
  for class_folder in os.listdir(sel_path):
    class_folder_path = os.path.join(sel_path, class_folder)

    # check if it's an actual directory (a class folder)
    if os.path.isdir(class_folder_path):
      # count number of image files in the class folder  
      image_files = [file for file in os.listdir(class_folder_path)]
      num_images = len(image_files)

      # store count in directory
      image_counter[class_folder] = num_images
  
  # # print the number of images in each class folder
  # for class_name, count in image_counter.items():
  #   print(f"Class ({class_name}): {count} images ")
  
  return image_counter
 


# Example usage
# path = "D:/Machine_Learning/Portfolio_Project_Machine_Learning/Mobile_Phone_Recognition"
# img_file_list = os.listdir(path)
# rename_images(path, img_file_list)