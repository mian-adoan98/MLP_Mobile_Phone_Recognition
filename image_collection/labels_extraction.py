# scraping labels from orange mobile phone websites
# scraping 1000 labels from weblinks
import os 
import sys
import requests
from bs4 import BeautifulSoup
from tools_scraping_data import extract_mobile_labels

# --------------------- Functions ---------------------------------------------

# open file of amazon urls 
def copy_weblinks(in_filename: str) -> list: 
  with open(in_filename, "r") as infile: 
    weblinks = []

    # implement interation over weblinks 
    for link in infile:
      weblinks.append(link)
  return weblinks

# choose resource_text3.txt as standard file for webscraping
# extract all mobile labels from selected weblinks

def select_links_label(weblinks: list) -> tuple:
  # define list of mobile phones labels 
  mp_labels2D = []
  failed_links = []

  # extract labels from selected weblinks
  for link in weblinks:
    mp_labels1D, failed_link = extract_mobile_labels(link)   # return: [labels, failed_list]
    mp_labels2D.append(mp_labels1D)

    if failed_link:
      failed_links.append(failed_link)  
  all_labels = [label for label_list in mp_labels2D for label in label_list]
  return all_labels, failed_links

# save mobile phone labels in file
def save_mobile_labels(filename: str, mobile_labels: list):
  # open file to write all the labels in 
  with open(filename, "w", encoding="utf-8") as outfile:
    for label in mobile_labels:
      outfile.write(f"{label}\n")
  print(f"Labels succesfully written to {filename}")


# ------------------------ ----------------------- Instances -----------------------------------------------------------

# create a path to extract weblinks after function call
filenames = os.listdir("resources/")
folder = "resources/"
paths = [os.path.join(folder, filename) for filename in filenames if "resource_page" in filename]
# print(f"Path list: {paths}")

# identify number of weblinks per page
for i in range(len(paths)):
  weblinks = copy_weblinks(paths[i])
  num_weblinks = len(weblinks)
  print(f"Filename ({paths[i]}) --> number of weblinks: {num_weblinks}")

# identify how many labels and failed links 
# mp_labels, failed_links = select_links_label(weblinks=weblinks[:10])                # 150 labels (10 links)
mp_labels, failed_links  = select_links_label(weblinks=weblinks[:50])                 # 786 labels
print(f"Number of labels: {len(mp_labels)} labels") 
print(f"Failed links: {len(failed_links)}")
print(failed_links)

# save labels in mobile_label_file.txt
mobile_label_file = "image_collection\\mobile_labels2.txt"
save_mobile_labels(mobile_label_file, mp_labels)

# ------------------------------------------------------------------------------------------------------


