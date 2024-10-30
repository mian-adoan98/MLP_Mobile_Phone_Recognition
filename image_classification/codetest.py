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

unknown_links = []
for link in link_lst1:
  if link.startswith("//"): 
    unknown_links.append(link)

print(len(unknown_links))