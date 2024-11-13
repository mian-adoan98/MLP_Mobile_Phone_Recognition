# Image processing: Function Declaration script

import os 
import pandas as pd 
import numpy as np 


# implement function 4: construct a column list
def construct_dict(features: list, data: list) -> dict: 
  web_content = {feature: sample for feature, sample in zip(features, data)}
  return web_content

# implement function 3: extracting labels
def extracting_labels(filename: str)-> np.ndarray:
  # define list --> extracting labels
  label_list = []

  # open file --> iterate the labels
  with open(filename, "r") as infile:
    for label in infile: 
      label_list.append(label)
  # convert to numpy array
  label_array = np.array(label_list)
  return label_array

