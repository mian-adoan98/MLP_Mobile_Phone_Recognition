# download images
from selenium import webdriver
from selenium.webdriver.common.by import By 
import time 
from tools_scraping_data import  download_images, extract_mobile_images
# scrape_mobile_images,
import numpy as np 

# ------------------------ functions --------------------------------
# extract weblinks from loaded resource file
def extract_weblinks(filename:str) -> list: 
  # define list --> store the weblink
  weblink_list = []
  # open the resource file
  with open(filename, "r") as infile:
    for link in infile: 
      weblink_list.append(link)
  return weblink_list

# # implement function: select a defined list of weblinks + download images
# def select_links_download(links: list, folder_dir: str):
#   # Creating link batch: store for every k link in n images
#   img_batched_coll = []
#   img_coll = {}
#   # Download images from selected links
#   for i,url in enumerate(links): 
#     image_urls = extract_mobile_images(url)
#     img_batched_coll.append(image_urls)
#     img_coll[f"Link {i}"] = img_batched_coll

#   # Compute the number of images per link
#   num_images = sum([len(lst) for lst in img_batched_coll])
#   print(f"Number of images for download: {num_images} images")

#   # Combine all image URLs into a single list for downloading
#   all_image_urls = np.array([url for sublist in img_batched_coll for url in sublist])

#   # Donwload all images from image URLs
#   download_images(all_image_urls, folder_dir)

# ------------------------- Instances -------------------------------------

# File directories where image are being downloaded
train_folder_dir = "mobile_phone_images\\train_images"
test_folder_dir = "mobile_phone_images\\test_images"
validation_folder_dir = "mobile_phone_images\\validation_images"

# load the file of weblinks
resource_dir = "D:\\Machine_Learning\\Portfolio_Project_Machine_Learning\\Mobile_Phone_Recognition\\resources\\resource_page3.txt"
amazon_links = extract_weblinks(resource_dir)

# function 2 -> extract mobile images from web  
def img_url_extractor(sel_links:list) -> np.ndarray:
  image_urls2d = []
  for i, weblink in enumerate(sel_links):
    
    image_urls = extract_mobile_images(weblink)
    image_urls2d.append(image_urls)
    print(f"weblink {i} is successfully processed\n")

  # convert into 1D-numpy arrays
  image_urls2d = np.array(image_urls2d, dtype=object)
  all_images = np.concatenate(image_urls2d)  # Flatten
  return all_images

all_images = img_extractor(sel_links=amazon_links[:50])
num_imgs = all_images.shape[0]
print(f"Number images: {num_imgs}")

# # download images into mobile image folder 
# mp_image_folder = "D:\\Machine_Learning\\Portfolio_Project_Machine_Learning\\Mobile_Phone_Recognition\\mobile_phone_images\\mobile_images"
# mp_image_folder2 = "D:\\Machine_Learning\\Portfolio_Project_Machine_Learning\\Mobile_Phone_Recognition\\mobile_phone_images\\mobile_images_v3"
# select_links_download(weblinks[:50], mp_image_folder2)