# scraping labels from orange mobile phone websites

import os 
import sys
import requests
import pandas 
from bs4 import BeautifulSoup
# from webscraping import webscraper_tool_man
from image_classification.tools_scraping_data import Webscraper, Images, Labels

# load website urls 
def load_web_urls(filename): 
  # create list for saving web url
  web_url_list = []
  with open(filename, "r") as url_file: 
    for line in url_file: 
      web_url_list.append(line)
  
  return web_url_list

weblinks_list = load_web_urls()
print(weblinks_list)

# ## Orange_url --- --------------------------------------------------------------------------------------------------------------
# # define the requirements for extracting labels from web data
# orange_response = requests.get(orange_url)

# # create webscraper object --> check the status, create a parser
# orange_scraper = Webscraper(orange_response)
# # orange website
# orange_html = amazon_scraper.check_status_request()
# amazon_soup = amazon_scraper.create_parser(orange_html)

# # print(amazon_soup)

# # create image object --> extract images from html page
# all_images = amazon_soup.find_all("img")
# image_extractor = Images(all_images)
# images_urls = image_extractor.extract_images(orange_url)
# image_df = image_extractor.store_image_df(images_urls)

# # create a labels object --> extract label from html page
# all_labels = amazon_soup.find_all("span")
# label_extractor = Labels(all_labels)
# print(all_labels)


# # save the image data as a csv_file format
# save_file = lambda filename, df: df.to_csv(filename)
# save_file("image_data2.csv", image_df)

# print(sys.path)

## Amazon URL 1 -------------------------------------------------------------------------------------------------------------------
def convert_url_to_html(url): 
  # create object for converting url to html
  webscraper = Webscraper(response)



  # define a webscraper object + create response --> extract data from aamazon website
  response = requests.get(amazon_url1)

  # create webscraper object --> check the status, create a parser
  
  # website
  html_content = amazon_scraper.check_status_request()
  websoup = amazon_scraper.create_parser(html_content)

  # amazon website
  html_content = amazon_scraper.check_status_request()
  return html_content

# print(websoup)

def webscrape_images(url): 
  # create image object --> extract images from html page
  all_images = amazon_soup.find_all("img")
  image_extractor = Images(all_images)
  images_urls = image_extractor.extract_images(amazon_url1)
  image_df = image_extractor.store_image_df(images_urls)

  # create a labels object --> extract label from html page
  all_labels = amazon_soup.find_all("span")
  label_extractor = Labels(all_labels)
  print(all_labels)

  # save the image data as a csv_file format
  save_file = lambda filename, df: df.to_csv(filename)
  save_file("image_data2.csv", image_df)

  print(sys.path)


# # ERROR : 
# # 
# # - TypeError: Images.store_image_df() takes 1 positional argument but 2 were given (line 28)
# # #