## Image Data Collection 
# import dependencies for system configuration
import os
import sys
import warnings
import time

# import dependencies for collecting images
import pandas as pd 
import numpy as np 

# Import dependencies for extracting images
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By 

## ---------------------------------------------------- Images -------------------------------------------------------------
## Help Function Declaration -----------------------------------------------------------------------------

# Function 1: Loading dataset
def amazon_data_loading(filename: str, n: int) -> pd.DataFrame:
    # Load amazon link dataset
    amazon_ds = pd.read_csv(filename)
    if n <= 10 or n > 0: 
        amazon_ds_sel = amazon_ds.head(n)
        return amazon_ds_sel
    elif n > 10:
        return amazon_ds
    

# Help function 2: Setting up WebDriver
def setup_webdriver_env() -> webdriver.edge.webdriver.WebDriver:
    # Setup environment for WebDriver on Edge Browser
    options = Options()
    options.add_argument("--headless")

    # Setup WebDriver service 
    edgedriver_path = "D:\\Data_Engineering\\data_extraction\\msedgedriver.exe"
    service = Service(executable_path=edgedriver_path)

    driver = webdriver.Edge(options=options, service=service)
    return driver




# Main Function Declaration ------------------------------------------------------------------------------------------------
# Function 3: Extracting images dataset
def image_extraction(filename: str, class_attr: str = "s-image") -> list:
    # Retrieve dataset and webdriver 
    driver = setup_webdriver_env()
    amazon_ds = amazon_data_loading(filename)

    # Iterate over a list of links --> extract images 
    item_img_lst = []
    items_tot = 0
    for i, link in enumerate(amazon_ds["links"].values[0:3]):
        # Retrieve image elements from each web URL    
        webdriver.get(link)
        img_elements = webdriver.find_elements(By.CLASS_NAME, class_attr)
        
        # Extract images from each amazon weblink + determine the number of images per url
        img_seq = [web_elem.get_attribute("src") for web_elem in img_elements]
        seq_size = len(img_seq)

        # Store image items into a list + determine the total number of images for the entire list
        item_img_lst.append(img_seq)
        items_tot += seq_size
        print(f"Amazon URL {i + 1}: {seq_size} items (successfully extracted)")
    print(f"Image Extraction is completed. Number of Image URLs: {items_tot} ")

    # Create a DataFrame with image items
    img_list = [item for item_seq in item_img_lst for item in item_seq]
    img_dataset = pd.DataFrame()
    img_dataset["Image_URL"] = pd.Series(img_list)
    return img_dataset

def label_extraction(filename):
    # Retrieve dataset and webdriver 
    driver = setup_webdriver_env()
    amazon_ds = amazon_data_loading(filename)

    # Iterate over list of amazon links: extract item titles 
    item_title_lst = []
    item_tot = 0
    for i, link in enumerate(amazon_ds["links"].values): 
        # Retrieve URL from dataset
        webdriver.get(link)
        time.sleep(5)

        # Extract the product titles from each link + take the size of each sequence
        text_sequence = driver.execute_script(""" 
            var h2_selector = "h2.a-size-base-plus.a-spacing-none.a-color-base.a-text-normal";
            var titles = [];
            var elements = document.querySelectorAll(h2_selector);
            elements.forEach(function(element){
                titles.push(element.innerText);                                       
            });
            return titles
        """)
        seq_len = len(text_sequence)

        # Store text sequence in list of all item titles
        item_title_lst.append(text_sequence)
        print(f"Amazon URL {i}: {seq_len} items (extracted successfully) ")
        item_tot += seq_len
    print(f"Extraction Successful. Number of Items: {item_tot}")

    # Convert into DataFrame
    item_titles = np.array([item for title_seq in item_title_lst for item in title_seq])
    labels_dataset = pd.DataFrame()
    labels_dataset["Labels"] = pd.Series(labels_dataset)
    
    return labels_dataset

