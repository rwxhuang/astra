import pandas as pd
import streamlit as st
import time

from abc import ABC, abstractmethod
from selenium import webdriver
from selenium.webdriver.common.by import By
from st_files_connection import FilesConnection
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from utils.data_utils import extract_tx_level, encode_locations, encode_status, normalize_views
from sklearn.preprocessing import MinMaxScaler


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


class Dataset(ABC):
    """
    Interface for handling datasets.
    Defines the required methods for any dataset handler.
    """
    @abstractmethod
    def load_data(self):
        """Load the data from the S3 bucket."""
        pass

    @abstractmethod
    def load_processed_data(self):
        """Process the data and return it."""
        pass


class TechportData(Dataset):
    """
    A class for loading and processing data from the Techport dataset.

    Attributes:
        conn (object): The connection object used to access the data.
        bucket (str): The name of the S3 bucket where the data is stored.
        file_name (str): The name of the CSV file to load.
    """
    _instance = None

    def __init__(self, conn):
        self.conn = conn
        self.bucket = "astra-data-bucket"
        self.file_name = "load_data_after_scrape.csv"

    def load_data(self):
        '''
        data for users in streamlit from techport scraped data
        '''
        with self.conn.open(self.bucket + '/' + self.file_name, "rb", encoding='utf-8') as f:
            df = pd.read_csv(f)
        df.set_index('PROJECT_ID', inplace=True)

        # make sure date cols are datetime types
        date_time_cols = ['START_DATE', 'END_DATE', 'LAST_MODIFIED']
        for col in date_time_cols:
            df[col] = pd.to_datetime(df[col])

        # make tx level readable
        df = extract_tx_level(df)

        return df

    def load_processed_data(self):
        '''
        process data for backend use
        '''
        df = (
            self.load_data()
            .pipe(encode_locations)
            .pipe(encode_status)
            .pipe(normalize_views, 0, 1)
        )

        return df


class SBIRData(Dataset):
    """
    A class for loading and processing data from the SBIR dataset.

    Attributes:
        conn (object): The connection object used to access the data.
        bucket (str): The name of the S3 bucket where the data is stored.
        file_name (str): The name of the CSV file to load.
    """

    conn = st.connection('s3', type=FilesConnection)

    def __init__(self, conn):
        self.conn = conn
        self.bucket = "astra-data-bucket"
        self.file_name = "sbir_no_abstract_all.csv"

    def load_data(self):
        df = self.conn.read(self.bucket + '/' + self.file_name,
                            input_format="csv", ttl=600)
        # Clean data
        # column names (all uppercase + connect with underscores)
        df.columns = df.columns.str.upper().str.replace(' ', '_')
        # AWARD_AMOUNT column into float values
        df['AWARD_AMOUNT'] = df['AWARD_AMOUNT'].astype(str).replace(
            ',', '').astype(float)
        return df

    def load_processed_data(self):
        df = self.load_data()

        # Normalize columns
        normal_columns = ['AWARD_AMOUNT', 'NUMBER_EMPLOYEES']
        scaler = MinMaxScaler()
        df[normal_columns] = scaler.fit_transform(df[normal_columns])

        # Encode columns to binary
        binary_columns = ['SOCIAL_DISADVANTAGED', 'WOMEN_OWNED']
        df[binary_columns] = df[binary_columns].apply(
            lambda x: 1 if x == 'Y' else 0)

        return df


@st.cache_data
def get_astra_data(search_input):
    """
    Returns a pandas DataFrame based on the search_input by web-scraping from Techport.

    Returns:
        pd.DataFrame: 
    """
    # Fetch Techport data from S3
    conn = st.connection('s3', type=FilesConnection)
    # TODO: merge TechportData(conn).load_data() with SBIRData(conn).load_data()
    techport_and_sbir = pd.merge(TechportData(conn).load_data(), SBIRData(
        conn).load_data(), on=['TITLE', 'START_YEAR', 'END_YEAR'], how='left')
    if not search_input:
        return techport_and_sbir
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
    filtered_df = techport_and_sbir.loc[list(project_ids)]
    return filtered_df
