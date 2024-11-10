# download images
from tools_scraping_data import scrape_mobile_images, download_images
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
    

# implement function: select a defined list of weblinks + download images
def select_links_download(links: list, folder_dir: str):
  # Creating link batch: store for every k link in n images
  img_batched_coll = []
  img_coll = {}
  # Download images from selected links
  for i,url in enumerate(links): 
    image_urls = scrape_mobile_images(url)
    img_batched_coll.append(image_urls)
    img_coll[f"Link {i}"] = img_batched_coll

  # Compute the number of images per link
  num_images = sum([len(lst) for lst in img_batched_coll])
  print(f"Number of images for download: {num_images} images")

  # Combine all image URLs into a single list for downloading
  all_image_urls = [url for sublist in img_batched_coll for url in sublist]

  # Donwload all images from image URLs
  download_images(all_image_urls, folder_dir)

# ------------------------- Instances -------------------------------------

# # load file of all labels
# file = load_file("image_collection\\mobile_labels.txt")
# print(file)

# File directories where image are being downloaded
train_folder_dir = "mobile_phone_images/train_images"
test_folder_dir = "mobile_phone_images/test_images"
validation_folder_dir = "mobile_phone_images/validation_images"

# load the file of weblinks
resource_dir = "D:\Machine_Learning\Portfolio_Project_Machine_Learning\Mobile_Phone_Recognition\\resources\\resource_page3.txt"
weblinks = extract_weblinks(resource_dir)
# for i, weblink in enumerate(weblinks):
#   print(f"{i} - {weblink}")

# download images into mobile image folder 
mp_image_folder = "D:\Machine_Learning\Portfolio_Project_Machine_Learning\Mobile_Phone_Recognition\mobile_phone_images\mobile_images"
select_links_download(weblinks[:50], mp_image_folder)