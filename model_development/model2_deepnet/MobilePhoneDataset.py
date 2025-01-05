import os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image

# Import dependencies from PyTorch
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

# Class: MobilePhoneDataset
# Define class Dataset: reconstruct images and labels into a new dataset
class MobileImageDataset(Dataset):
  def __init__(self, csv_file, data_dir, transform=None):
    # Reconstruct dataset
    self.transform = transform 
    self.data_dir = data_dir 
    self.csv_file_dir = os.path.join(self.data_dir, csv_file)
    self.data_name = pd.read_csv(self.csv_file_dir)
    
    # Length of dataset
    self.len = self.data_name.shape[0]

  def __len__(self):
    return self.len
  

  def __getitem__(self, idx):
    # Load the image from dataset
    data_name = os.path.join(self.data_dir, self.data_name.iloc[idx, 2])
    mobile_image = Image.open(data_name)
    ybrand = self.data_name.iloc[idx, 1]
    
    # Transform the original mobile_image, if transformer exists
    if self.transform:
      mobile_image = self.transform(mobile_image)
    return mobile_image, ybrand
  
# Define Transformations    (CHECK Coursera: link --> https://www.coursera.org/projects/deep-learning-pytorch/peer/3Q7Q1/deep-learning-with-pytorch)
class TransformMobileImage:
  def __init__(self):
    self.data_transform = transforms.Compose([
      transforms.Resize((128, 128)),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
  def transform_image(self, img):
    return self.data_transform(img)


