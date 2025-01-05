import pandas as pd
import streamlit as st
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from st_files_connection import FilesConnection
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


class TechportScraper:
    def __init__(self, search_input):
        """
        Initialize the scraper with a search_input.

        :param search_input: The search input for the Techport website
        """
        self.search_input = search_input

        options = Options()
        options.add_argument("--headless")
        self.driver = webdriver.Chrome(service=Service(
            ChromeDriverManager().install()), options=options)

    def _wait_by_class(self, class_name):
        WebDriverWait(self.driver, 5).until(
            EC.presence_of_element_located(
                (By.CLASS_NAME, class_name))
        )

    def run(self):
        """
        Main method to coordinate the scraping process.
        """
        # Open website
        self.driver.get("https://techport.nasa.gov/")
        # Input search_input
        self._wait_by_class('_searchField_3htcq_44')
        time.sleep(1)
        self.driver.find_element(By.CLASS_NAME, '_searchField_3htcq_44') \
            .find_element(By.TAG_NAME, 'input') \
            .send_keys(self.search_input)
        self.driver.find_element(
            By.CLASS_NAME, '_searchButton_3htcq_82').click()
        # Find all the project ids
        project_ids = set()
        while True:
            self._wait_by_class('_pageNumberInput_bcc9b_35')
            for proj_desc in self.driver.find_element(By.TAG_NAME, 'tbody').find_elements(By.TAG_NAME, 'tr'):
                project_id = int(proj_desc.find_element(By.TAG_NAME, 'a').text)
                project_ids.add(project_id)
            if self.driver.find_element(By.CLASS_NAME, '_pageNumberInput_bcc9b_35').get_attribute('value') == self.driver.find_element(By.CLASS_NAME, '_pageNumberInput_bcc9b_35').get_attribute('max'):
                break
            self.driver.find_element(
                By.CLASS_NAME, '_forward_w82bl_70').click()
        return project_ids


@st.cache_data
def get_astra_data(search_input):
    """
    Returns a pandas DataFrame based on the search_input by web-scraping from Techport.

    Returns:
        pd.DataFrame: 
    """
    # Fetch Techport data from S3
    conn = st.connection('s3', type=FilesConnection)
    df = conn.read("astra-data-bucket/techport_121824_clean.csv",
                   input_format="csv", ttl=600)
    df.set_index('PROJECT_ID', inplace=True)
    if not search_input:
        return df
    # Get Techport project ids
    scraper = TechportScraper(search_input)
    attempts = 0
    while attempts < 3:
        try:
            project_ids = scraper.run()
            break
        except TimeoutException:
            attempts += 1
            st.error(
                "The scraper took too long to run and couldn't find any results. Please try again.")
            project_ids = []
    # Filter the data
    filtered_df = df.loc[list(project_ids)]
    return filtered_df
