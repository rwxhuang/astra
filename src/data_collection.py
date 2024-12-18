import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By


class TechportScraper:
    def __init__(self, search_input):
        """
        Initialize the scraper with a search_input.

        :param search_input: The search input for the Techport website
        """
        self.search_input = search_input
        self.driver = webdriver.Chrome()

    def run(self):
        """
            Main method to coordinate the scraping process.
        """

        self.driver.get("https://techport.nasa.gov/advancedSearch")
        search_button = self.driver.find_element(
            By.CLASS_NAME, '_large_s5jgh_128')
        search_button.click()


def get_astra_data(search_input):
    """
    Returns a pandas DataFrame based on the search_input by web-scraping from Techport.

    Returns:
        pd.DataFrame: 
    """
    return pd.DataFrame()
