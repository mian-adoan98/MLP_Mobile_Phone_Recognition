## Webscraping Tool maangement 
# 
# function 1: scraping images 
# function 2: scraping labels 
# function 3: scraping prices 
# ... 
# #

from playwright.sync_api import sync_playwright
import pandas as pd 
import os 
import requests

# implement function for scraping images 
# implement function 1: extracting images from url (CHECK)

def scrape_mobile_images(url: str) -> list: 
  with sync_playwright() as p: 
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    # navigate to page with images
    page.goto(url)
    # exctract all images from selected page
    image_elements = page.query_selector_all('img')

    # collect image urls
    image_urls = [img.get_attribute("src") for img in image_elements if img.get_attribute("src")]
    return image_urls
    browser.close()

# implement function 2 -> scraping labels from images 
def scrape_mobile_labels(url: str) -> list: 
  with sync_playwright() as p: 
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    # navigate to page with images
    page.goto(url)
    # exctract all images from selected page
    label_elements = page.query_selector_all('span')

    # collect image urls
    label_urls = [label.get_attribute("class") for label in label_elements if label.get_attribute("class")]
    return label_urls
    browser.close()

# implement function 3 --> scraping price labels from smartphones
def scrape_mobile_labels(url: str) -> list: 
  with sync_playwright() as p: 
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    # navigate to page with images
    page.goto(url)
    # exctract all images from selected page
    price_elements = page.query_selector_all('span')

    # collect image urls
    price_urls = [price.get_attribute("class") for price in price_elements if price.get_attribute("class")]
    return price_urls
    browser.close()

# implement function 4: converting image urls into pandas dataframe (CHECK) (!!MOD)
def url_dataframe(image_url): 
  image_dict = {}
  image_dict["Image_URL"] = image_url 
  image_df = pd.DataFrame(image_dict)
  return image_df 

# implement function 5: downloading images from image urls after extracting it from weblink (CHECK)
def download_images(image_urls: list, 
                    image_folder_dir: str): 
  try: 
    # create a directory to save images
      os.makedirs(image_folder_dir, exist_ok=True)
  except FileExistsError as file_exist_error:
     print(f"Type of Error: {FileExistsError}")
     print(f"Description: {file_exist_error}")
     print(f"Filename: {file_exist_error.filename}")

  # implement iteration: download each image  
  for i, url in enumerate(image_urls):
    if not url.startswith(("http://", "https://")):
      print(f"Skipping invalid URL: {url}")
      continue 
    
    try:
        # Fetch the image
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP errors

        # Create a file name based on the image URL
        file_extension = url.split('.')[-1]
        filename = f"image_{i + 1}.{file_extension}"
        filepath = os.path.join(image_folder_dir, filename)

        # Save the image
        with open(filepath, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        
        print(f"Downloaded: {filename}")

    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")
        # image_urls.remove(url)

## Problem 
# 
# function 5: unknown url causes infinite looping --> remove unknown url# 

# function  5: removing invalid urls
def remove_invalid_urls(filename: str) -> list: 
  # define 2 validated prefixes: http:// and https://
  valid_url_prefix = ("http://","https://")

  # open file: look for invalid urls for removal
  with open(filename, "r") as infile: 
    urls = [url.strip() for url in infile.readlines()]

  # filter valid urls 
  validate_urls = [url.strip() for url in urls if url.strip().startswith(valid_url_prefix)]
  # overwrite current file by new list of urls
  with open(filename, "w") as outfile:
    for url in validate_urls:
      outfile.write(url + "\n")

  # compute number of valid and invalid urls
  num_val_url = len(validate_urls)
  num_inval_url = len(urls) - len(validate_urls)
  print(f"Removed {len(urls) - len(validate_urls)} invalid urls")
  return num_val_url, num_inval_url
  return validate_urls


# function 6: validate urls --> valid or invalid (CHECK)
def validate_urls(filename:str) -> tuple:
  # define list of urls: validated or invalidated
  validate_urls = []
  invalidated_urls = []
  num_val_url, num_inval_url = (0,0)
  # define prefix set for validated urls
  valid_url_prefix = (("http://","https://"))

  # open file --> check every url (valid + invalid)
  with open(filename, "r") as infile:
    for weburl in infile:
      weburl.strip()  # removing whitespaces
      if not weburl.startswith(valid_url_prefix): 
        invalidated_urls.append(weburl)
        num_inval_url += 1
      else:
        validate_urls.append(weburl)
        num_val_url += 1
    
    return num_val_url, num_inval_url
          

   