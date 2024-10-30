# task 1: image collection 
from tools_scraping_data import scrape_mobile_images, remove_invalid_urls
from tools_scraping_data import download_images, validate_urls
import os

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
paths = [os.path.join(folder, filename) for filename in filenames]
print(f"Path list: {paths}")

link_lst1 = copy_weblinks(paths[0]) # 250
link_lst2 = copy_weblinks(paths[1]) # 13 
link_lst3 = copy_weblinks(paths[2]) # 52

# validate link page: 
for i in range(len(paths)):
  val_urls, inval_urls = validate_urls(paths[i])
  print(f"Filename {i+1}: {paths[i]}")
  print(f"Valid URLs: {val_urls}")
  print(f"Invalid URLs: {inval_urls}\n")

# remove invalid weblink
# num_val_url, num_inval_url = remove_invalid_urls(paths[1])
# print(f"Valid URL (after removal): {num_val_url}")
# print(f"Invalid URL (after removal): {num_inval_url}")

# compute the number of image urls per page of weblinks (!!)
total_img_urls = 0

# for i, link in enumerate([link_lst1[:10], link_lst2[:10], link_lst3[:10]]):
#   print(f"Number of links {i+1}: {len(link)} links")
#   for web_url in link: 
#     image_urls = scrape_mobile_images(web_url)
#     num_images = len(image_urls)
    
#   print(f"Link list {i}: {num_images} images")
  
  
# # implement batch sizing: split links into batches
# batch_nr = 50
# links1 = link_lst1[:batch_nr]
# link_lst2 # create link_lst3 for downloading images 
test_folder_dir = "mobile_phone_images/test_images"

# implement function: select a defined list of weblinks + download images
def select_link_download(links: list, folder_dir: str):
  # creating link batch: store for every k link in n images
  img_batched_coll = []
  link_img_coll = {}
  # download images from selected links
  for i,url in enumerate(links): 
    image_urls = scrape_mobile_images(url)
    img_batched_coll.append(image_urls)
    link_img_coll[f"Link {i}"] = img_batched_coll

  return link_img_coll 
    # download_images(images_urls, folder_dir)

link_batch_images = select_link_download(links=link_lst1[:2], folder_dir=test_folder_dir)
print(link_batch_images)
# select_link_download(links=links12, folder_dir=test_folder_dir)
# select_link_download(links=links13, folder_dir=test_folder_dir)




