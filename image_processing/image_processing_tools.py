# Image processing: Function Declaration script

import os 
import glob
import shutil
import pandas as pd 
import numpy as np 

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

# implement function 3 --> move images to other (version 2) (CONTINUE WORKING)(!!)
def distribute_images(in_directory: str, 
                      out_directory: str, 
                      dataframe: pd.DataFrame):
  # extract class folders into list: list
  # pair the image item with its label: dict
  # implement iteration: rearrange and distribute the images based on
  # -> create the path name by joining image name
  # -> #

# Example usage
# path = "D:/Machine_Learning/Portfolio_Project_Machine_Learning/Mobile_Phone_Recognition"
# img_file_list = os.listdir(path)
# rename_images(path, img_file_list)