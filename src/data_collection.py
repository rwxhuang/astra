import pandas as pd
import streamlit as st
import time
import numpy as np

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
        with self.conn.open(self.bucket + '/' + self.file_name, "rb", encoding='utf-8') as f:
            df = pd.read_csv(f)
        df.set_index('PROJECT_ID', inplace=True)
        
        # make sure date cols are datetime types
        date_time_cols = ['START_DATE', 'END_DATE', 'LAST_MODIFIED']
        for col in date_time_cols:
            df[col] = pd.to_datetime(df[col])

        # make tx levels readable
        def extract_tx_level(dataframe):
            '''
            dataframe: take in a dataframe
            return a dataframe with the extracted tx level
            '''
            # collect first and second parts of tx level
            dataframe['TX_PART1'] = dataframe['PRIMARY_TX'].str.extract(r'TX(\d{2})')
            dataframe['TX_PART2'] = dataframe['PRIMARY_TX'].str.extract(r'TX\d{2}\.(\d+|X)')
            dataframe['TX_PART2'] = dataframe['TX_PART2'].replace('X', '0').fillna('0')

            # combine parts into a float for extracted tx
            dataframe['TX_EXTRACTED'] = dataframe['TX_PART1'] + '.' + df['TX_PART2']
            dataframe['TX_EXTRACTED'] = dataframe['TX_EXTRACTED'].astype(float)

            dataframe = dataframe.drop(columns=['TX_PART1', 'TX_PART2'])

            return dataframe
        
        df = extract_tx_level(df)

        return df

    def load_processed_data(self):
        df = self.load_data()
        
        def encode_locations(dataframe):
            '''
            dataframe: given a dataframe
            returns a dataframe with col for locations encoded
            '''
            # get all different combinations of locations listed
            locations_list = dataframe['LOCATIONS_WHERE_WORK_IS_PERFORMED'].unique()

            unique_locations = set()

            # for each combination
            for elem in locations_list:
                # if string
                if isinstance(elem, str):
                    # try to split it 
                    locations = elem.split('; ')
                    # add each split location
                    for location in locations:
                        unique_locations.add(location)  # Add each location to the set
                # handles NaN's
                else:
                    continue

            unique_locations.remove('Not Applicable')
            unique_locations.remove('Outside the United States ')

            # mapping from location to index
            location_to_index = {location: index for index, location in enumerate(unique_locations)}
            location_to_index['Outside the United States '] = location_to_index['Outside the United States']


            def convert_to_vector_locations(locations):
                '''
                Takes in locations with form state; state; ...
                and converts it to a one-hot encoding
                '''
                # initialize binary vector
                vector = np.zeros(len(unique_locations), dtype=int)

                # if locations is NaN or N/A then don't update vector
                if not isinstance(locations, str) or locations == 'Not Applicable':
                    pass

                else:
                    # get every individual location
                    location_list = locations.split('; ')
                    
                    # for each location, update it's spot in the vector to 1
                    for loc in location_list:
                        vector[location_to_index[loc]] = 1
                
                return vector

            dataframe['LOCATIONS_ENCODED'] = dataframe['LOCATIONS_WHERE_WORK_IS_PERFORMED'].apply(convert_to_vector_locations)

            return dataframe
        
        def encode_status(dataframe):
            '''
            dataframe: given a dataframe
            return the dataframe with a col for encoding for completed, active, or cancelled
            '''

            statuses = {'Active', 'Completed', 'Canceled'}

            status_to_index = {status: index for index, status in enumerate(statuses)}

            def convert_to_vector_statuses(status):
                '''
                one hot encodes the given status
                '''
                vector = np.zeros(len(status_to_index), dtype=int)
                vector[status_to_index[status]] = 1
                return vector
            
            dataframe['STATUS_ENCODED'] = dataframe['STATUS'].apply(convert_to_vector_statuses)
            
            return dataframe
        
        def normalize_views(dataframe, lower, upper):
            '''
            dataframe: takes in a dateframe
            lower: lower bound min
            upper: upper bound max
            returns a dataframe with the views normalized from lower to upper
            '''

            scaler = MinMaxScaler(feature_range=(lower,upper))
            dataframe['VIEW_COUNT_NORMALIZED'] = scaler.fit_transform(dataframe[['VIEW_COUNT']])
            return df

        df = encode_locations(df)
        df = encode_status(df)
        df = normalize_views(df, 0, 1)

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
        self.file_name = "sbir_all.csv"

    def load_data(self):
        df = self.conn.read(self.bucket + '/' + self.file_name,
                            input_format="csv", ttl=600)
        return df

    def load_processed_data(self):
        df = self.load_data()
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
    df = TechportData(conn).load_data()
    # TODO: merge TechportData(conn).load_data() with SBIRData(conn).load_data()
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
