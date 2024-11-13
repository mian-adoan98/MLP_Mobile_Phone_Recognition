# Image processing tools: Function Declaration script
# 
# purpose: effective and qualitative processing of images striving at 100% accuracy
# function 1: constructing a single column series
# function 2: extracting labels #
import os 
import pandas as pd 
import numpy as np 


# implement function 1: construct a column list
def construct_feature(feature_name: str, samples: list) -> pd.DataFrame:
  feature = pd.DataFrame()
  feature[feature_name] = pd.Series(samples)
  return feature

# implement function 2: rename the images from the selected folder -----> (NEED CORRECTION !!)

def rename_images(path: str, img_file_list: list):
  # first pass: rename files to temporary names to avoid overwriting
  for i, filename in enumerate(img_file_list):
    old_path = os.path.join(path, filename)         # original file path
    temp_filename = f"temp_{i}.jpg"                 # temporary filename to prevent conflicts
    temp_path = os.path.join(path, temp_filename)
    
    # Check if the old file exists
    if not os.path.exists(old_path):
      print(f"File not found: {old_path}")
      continue
    # rename to temprary filename
    os.rename(old_path, temp_path)   

  # update list of files after first renaming 
  temp_file_list = os.listdir(path)

  # second pass: rename files to final numeric names
  for i, temp_filename in enumerate(temp_file_list, start=1):
    temp_path = os.path.join(path, temp_filename)    # temporary file path 
    new_filename = f"mobile_phone_{i}.jpg"           # final desired filename (e.g., mobile_image_1.jpg, mobile_image_2.jpg, etc.)
    new_path = os.path.join(path, new_filename)

    # Check if the temp file exists before renaming
    if not os.path.exists(temp_path):
      print(f"Temp file not found: {temp_path}")
      continue
    # rename to final filename
    os.rename(temp_path, new_path)
    print()