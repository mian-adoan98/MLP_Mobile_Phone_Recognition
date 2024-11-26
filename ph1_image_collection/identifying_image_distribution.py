# task 1: image collection 
# describing distribution of images
from ph1_image_collection.data_extraction_tools import scrape_mobile_images, remove_invalid_urls
from ph1_image_collection.data_extraction_tools import validate_urls
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# fetch images 
async def fetch_images(session, link):
    return await scrape_mobile_images(link)  # Assume this is now an async function

# open file of amazon urls 
def copy_weblinks(in_filename: str) -> list: 
  with open(in_filename, "r") as infile: 
    weblinks = []

    # implement interation over weblinks 
    for link in infile:
      weblinks.append(link)
  return weblinks

# create a path to extract weblinks after function call
filenames = os.listdir("resources/")
folder = "resources/"
paths = [os.path.join(folder, filename) for filename in filenames if "resource_page" in filename]
print(f"Path list: {paths}")

# identify number of weblinks per page

for i in range(len(paths)):
  weblinks = copy_weblinks(paths[i])
  num_weblinks = len(weblinks)
  print(f"Filename ({paths[i]}) --> number of weblinks: {num_weblinks}")

# validate link page: 
for i in range(len(paths)):
  val_urls, inval_urls = validate_urls(paths[i])
  print(f"Filename {i+1}: {paths[i]}")
  print(f"Valid URLs: {val_urls}")
  print(f"Invalid URLs: {inval_urls}\n")

# remove invalid weblink
num_val_url, num_inval_url = remove_invalid_urls(paths[1])
print(f"Valid URL (after removal): {num_val_url}")
print(f"Invalid URL (after removal): {num_inval_url}")

# compute the number of image urls per page of weblinks (!!)
def identify_num_images(links: list, filename: str): 
    num_images = []
    
    with ThreadPoolExecutor() as executor:
        future_to_link = {executor.submit(scrape_mobile_images, link): link for link in links}
        
        for future in as_completed(future_to_link):
            image_urls = future.result()
            num_images.append(len(image_urls))

    tot_images = sum(num_images)
    print(f"{filename}: {tot_images} images")

weblinks1 = copy_weblinks(paths[2])
# identify_num_images(weblinks1[:10], "file1")          # 170 images (10 links)
identify_num_images(weblinks1[:50], paths[2])           # 779, 817, 851,823, 851 images 851 851 851
# identify_num_images(link_lst1[101:150], "file1")

