## Data Extraction Tools
# 
# purpose: extracting content from web resources 
#
# function 1: scraping images 
# function 2: scraping labels 
# function 3: scraping prices 
# ... 
# #

from playwright.sync_api import sync_playwright
from selenium import webdriver
from selenium.webdriver.common.by import By 
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import playwright
from bs4 import BeautifulSoup
import pandas as pd 
import os 
import requests
import time

# (WEBLINKS): implement function 0 --> extracting weblinks from resource files
# extract weblinks from loaded resource file
def extract_weblinks(filename:str) -> list: 
  # define list --> store the weblink
  weblink_list = []
  # open the resource file
  with open(filename, "r") as infile:
    for link in infile: 
      weblink_list.append(link)
  return weblink_list

# implement function for scraping images 
# (IMAGEs): implement function 1 --> extracting images from url (CHECK) (Option 1)

def scrape_mobile_images(url: str) -> list: 
  with sync_playwright() as p: 
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    # navigate to page with images
    try:
      page.goto(url, timeout=60000)
    except playwright._impl._errors.TimeoutError:
      print(f"Timeout exceeded while trying to load {url}")

    # exctract all images from selected page
    image_elements = page.query_selector_all('img')

    # collect image urls
    image_urls = [img.get_attribute("src") for img in image_elements if img.get_attribute("src")]
    browser.close()
    return image_urls

# (IMAGES) function 1 --> extract images from web links (Option 2)
def extract_mobile_images(weblink: str) -> list:
 # Set up Chrome options for headless mode and pop-up disabling
  chrome_options = Options()
  chrome_options.add_argument("--headless")  # Run in headless mode
  chrome_options.add_argument("--disable-notifications")
  chrome_options.add_argument("--disable-popup-blocking")
  # set up webdriver 
  driver = webdriver.Chrome(options=chrome_options)
  driver.get(weblink)

  # time.sleep(3)
  # Wait for the page to load (adjust based on internet speed)
  WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//img[@src]")))
  
  # locate all images elements
  images = driver.find_elements(By.XPATH, "//img[@src]")

  # extract image URLs
  image_urls = []

  for img in images: 
    src = img.get_attribute("src")
    if src and "media-amazon" in src: # only consider Amazon media URLS
      image_urls.append(src)

  # for idx, url in enumerate(image_urls):
  #   print(f"Image {idx + 1}: {url}")
  # close the browser 
  driver.quit()

  return image_urls

# (LABELs): implement function 2 -> scraping labels from URLs
mobile_labels = ["iphone", "samsung","oneplus","nokia","motorola", "xiaomi", "apple"]

def extract_mobile_labels(url: str) -> list: 
  # define list --> all smartphone company brands
  mobile_ = ["iphone", "samsung","oneplus","nokia","motorola", "xiaomi", "apple"]
  labels = []
  div_class = "a-section a-spacing-none a-spacing-top-small s-title-instructions-style"
  label_class = "a-size-base-plus a-color-base a-text-normal"
  
  # create a request --> asking web for allowance of extracting page content
  for attempt in range(3):
    response = requests.get(url)
    if response.status_code == 200:
      page_content = response.text
      # parse the content
      soup = BeautifulSoup(page_content, "html.parser")

      # extract label data from page content
      for item in soup.find_all("div", class_ = div_class):
        label_tag = item.find("span", class_ = label_class)
        label = label_tag.text.strip() if label_tag   else None 
        labels.append(label)
      break  # Exit loop if successful
    else:
      print(f"Failed to retrieve the Page. Status code: {response.status_code}")
      time.sleep(1)  # Wait 1 second before retrying
  return labels, url if not labels else None  # Return url if no labels are extracted


# (PRICE): implement function 3 --> scraping price labels from smartphones
def extract_mobile_price(url: str) -> list: 
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



## Problem 
# function 5: unknown url causes infinite looping --> remove unknown url# 

