## Image Data Collection : Object Oriented Programming
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

## ------------------------------------- Classes ---------------------------------------------------------------

class Extractor:
    def __init__(self, filename, content_type):
        self.filename = filename
        self.dataset = pd.read_csv(self.filename)
        self.class_attr = "s-image"
        self.content_type = content_type

        # Verify type of content to proceed
        if self.content_type == "Image":
            self.variable = "Image_URLs"
        elif self.content_type == "Label":
            self.variable = "Labels"
        else:
            print(f"Content Type unknown")

    # Transform items into dataset
    def transform(self, items: list):
        ds = pd.DataFrame()
        ds["Image_URL"] = pd.Series(items)

# Blueprint 1: ImageExtractor
class ImageExtractor(Extractor):
    def extract_sequences(self, webdriver: webdriver.edge.webdriver.WebDriver): 
        links = self.dataset["links"].values 
        for i, link in enumerate(links):
            # Retrieve image elements from each web URL    
            self.webdriver.get(link)
            img_elements = self.webdriver.find_elements(By.CLASS_NAME, self.class_attr)
        
            # Extract images from each amazon weblink + determine the number of images per url
            img_seq = [web_elem.get_attribute("src") for web_elem in img_elements]
            seq_size = len(img_seq)
        return img_seq
    
    def flatten(self, sequences: list[list]):
        image_items = np.array([item for seq in self.sequences for item in seq])
        return image_items
    

    
# Blueprint 2: LabelExtractor 
class LabelExtractor(Extractor):
    

    def extract_sequences(self, webdriver: webdriver.edge.webdriver.WebDriver): 
        # Define constant for extracting labels
        item_title_lst = []
        item_tot = 0

        # Iterate over links
        for i, link in enumerate(amazon_ds["links"].values): 
            # Retrieve URL from dataset
            webdriver.get(link)
            time.sleep(5)

            # Extract the product titles from each link + take the size of each sequence
            text_sequence = webdriver.execute_script(""" 
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
    
    
    def flatten(self, sequences: list[list]):
        image_items = np.array([item for seq in self.sequences for item in seq])
        return image_items
    
    def __str__(self):
        return print(f"Extraction Successful. Number of Items: {item_tot}")