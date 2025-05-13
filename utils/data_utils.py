
import numpy as np
import math
import re
import pandas as pd
import streamlit as st
from selenium import webdriver
from sklearn.preprocessing import MinMaxScaler
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.core.os_manager import ChromeType
from webdriver_manager.chrome import ChromeDriverManager

options = Options()
options.add_argument("--disable-gpu")
options.add_argument("--headless")


@st.cache_resource
def get_driver():
    return webdriver.Chrome(
        service=Service(
            ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install()
        ),
        options=options,
    )


def extract_tx_level(dataframe):
    '''
    dataframe: take in a dataframe
    return a dataframe with the extracted tx level
    TX8.1.1 --> 8.1
    TX8.1 --> 8.1
    TX8 --> 8.0
    TX8.X --> 8.0
    '''
    # collect first and second parts of tx level in temp cols
    dataframe['TX_PART1'] = dataframe['PRIMARY_TX'].str.extract(r'TX(\d{2})')
    dataframe['TX_PART2'] = dataframe['PRIMARY_TX'].str.extract(
        r'TX\d{2}\.(\d+|X)')

    # handles edge cases
    dataframe['TX_PART2'] = dataframe['TX_PART2'].replace('X', '0').fillna('0')

    # combine parts into a float for extracted tx
    dataframe['TX_EXTRACTED'] = dataframe['TX_PART1'] + \
        '.' + dataframe['TX_PART2']
    dataframe['TX_EXTRACTED'] = dataframe['TX_EXTRACTED'].astype(float)
    dataframe['TX_EXTRACTED_INT'] = dataframe['TX_EXTRACTED'].fillna(
        0).astype(int)
    # removes temp cols
    dataframe = dataframe.drop(columns=['TX_PART1', 'TX_PART2'])

    return dataframe


def encode_locations(dataframe):
    '''
    dataframe: given a dataframe
    returns a dataframe with col for locations encoded as a one-hot encoding
    '''

    # mapping from location to index
    location_to_index = {'Arkansas (AR)': 0, 'Minnesota (MN)': 1, 'West Virginia (WV)': 2,
                         'Maine (ME)': 3, 'Guam (GU)': 4, 'Alabama (AL)': 5, 'Louisiana (LA)': 6,
                         'Indiana (IN)': 7, 'Texas (TX)': 8, 'Connecticut (CT)': 9, 'Virginia (VA)': 10,
                         'Idaho (ID)': 11, 'Kansas (KS)': 12, 'Nevada (NV)': 13, 'Kentucky (KY)': 14,
                         'South Dakota (SD)': 15, 'Marshall Islands (MH)': 16, 'Oregon (OR)': 17,
                         'Michigan (MI)': 18, 'Washington (WA)': 19, 'Iowa (IA)': 20, 'Georgia (GA)': 21,
                         'District of Columbia (DC)': 22, 'Puerto Rico (PR)': 23, 'Utah (UT)': 24,
                         'Missouri (MO)': 25, 'Wyoming (WY)': 26, 'Outside the United States': 27,
                         'North Dakota (ND)': 28, 'New Mexico (NM)': 29, 'Arizona (AZ)': 30,
                         'Pennsylvania (PA)': 31, 'Montana (MT)': 32, 'Mississippi (MS)': 33,
                         'Wisconsin (WI)': 34, 'Alaska (AK)': 35, 'New Hampshire (NH)': 36,
                         'Delaware (DE)': 37, 'Colorado (CO)': 38, 'California (CA)': 39,
                         'Vermont (VT)': 40, 'Oklahoma (OK)': 41, 'Virgin Islands (VI)': 42,
                         'Tennessee (TN)': 43, 'Massachusetts (MA)': 44, 'Hawaii (HI)': 45,
                         'Maryland (MD)': 46, 'Ohio (OH)': 47, 'Florida (FL)': 48,
                         'South Carolina (SC)': 49, 'New York (NY)': 50, 'North Carolina (NC)': 51,
                         'Rhode Island (RI)': 52, 'Illinois (IL)': 53, 'Nebraska (NE)': 54,
                         'New Jersey (NJ)': 55, 'Outside the United States ': 27}

    def convert_to_vector_locations(locations):
        '''
        Takes in locations with form state; state; ...
        and converts it to a one-hot encoding [0,0,0,1,0...]
        '''
        # initialize binary vector, len - 1 because outside the US is repeated twice
        vector = np.zeros(len(location_to_index) - 1, dtype=int)

        # if locations is NaN or N/A then don't update vector
        if isinstance(locations, str) and locations != 'Not Applicable':
            # get every individual location
            location_list = locations.split('; ')

            # for each location, update it's spot in the vector to 1
            for loc in location_list:
                if loc:
                    vector[location_to_index[loc]] = 1

        return vector

    # apply function created to all entries in col
    dataframe['LOCATIONS_ENCODED'] = dataframe['LOCATIONS_WHERE_WORK_IS_PERFORMED'].apply(
        convert_to_vector_locations)

    return dataframe


def encode_status(df):
    '''
    df: given a dataframe
    return the dataframe with a col for encoding for completed, active, or cancelled
    '''
    df['STATUS_ENCODED'] = df['STATUS'].map(
        {'Active': 0, 'Completed': 1, 'Cancelled': 1})

    return df


def modify_views(df, lower, upper):
    '''
    dataframe: takes in a dateframe
    lower: lower bound min
    upper: upper bound max
    returns a dataframe with the views normalized from lower to upper
    '''

    # initialize scaler object with lower and upper bounds
    scaler = MinMaxScaler(feature_range=(lower, upper))

    # apply scaler to col
    df['VIEW_COUNT_NORMALIZED'] = scaler.fit_transform(
        df[['VIEW_COUNT']])
    df['LOG_VIEW_COUNT'] = np.log(df['VIEW_COUNT'] + 1)
    return df


def encode_tx_level(dataframe):
    '''
    dataframe: takes in a dataframe
    returns a dataframe with a col for encoding all 1 deep tx levels
    '''

    # mapping from level to index
    level_to_index = {14.1: 0, 12.5: 1, 6.2: 2, 1.2: 3, 9.5: 4, 17.2: 5,
                      1.1: 6, 9.4: 7, 1.4: 8, 8.1: 9, 5.3: 10, 12.1: 11,
                      3.1: 12, 15.1: 13, 11.5: 14, 17.0: 15, 4.3: 16, 15.2: 17,
                      3.2: 18, 9.2: 19, 7.2: 20, 5.5: 21, 9.1: 22, 11.6: 23,
                      4.4: 24, 12.3: 25, 7.1: 26, 12.4: 27, 17.4: 28, 4.2: 29,
                      14.2: 30, 10.2: 31, 6.4: 32, 17.3: 33, 5.2: 34, 4.6: 35,
                      6.3: 36, 5.6: 37, 14.3: 38, 8.2: 39, 11.1: 40, 5.1: 41,
                      5.4: 42, 6.5: 43, 16.3: 44, 2.1: 45, 1.3: 46, 12.2: 47,
                      16.5: 48, 3.3: 49, 2.2: 50, 11.4: 51, 4.1: 52, 13.2: 53,
                      6.1: 54, 17.1: 55, 4.5: 56, 10.3: 57, 10.1: 58, 16.4: 59,
                      8.3: 60, 10.4: 61, 11.2: 62, 16.2: 63, 17.5: 64, 13.1: 65,
                      2.3: 66, 6.6: 67, 11.3: 68, 7.3: 69, 13.4: 70, 17.6: 71,
                      16.6: 72, 13.3: 73, 16.1: 74, 9.3: 75, 8.0: 76, 3.0: 77,
                      4.0: 78, 1.0: 79, 12.0: 80, 2.0: 81, 9.0: 82, 11.0: 83,
                      5.0: 84, 6.0: 85, 14.0: 86, 15.0: 87, 16.0: 88, 7.0: 89,
                      10.0: 90, 13.0: 91}

    def convert_to_vector_levels(level):
        '''
        given a level, return a one-hot encoding of it
        '''

        # initialize empty binary vector
        vector = np.zeros(len(level_to_index), dtype=int)

        # if nan then skip and return empty vector
        if not math.isnan(level):
            vector[level_to_index[level]] = 1

        return vector

    # apply function created to all entries in col
    dataframe['TX_LEVEL_ENCODED_SUBLEVEL'] = dataframe['TX_EXTRACTED'].apply(
        convert_to_vector_levels)

    return dataframe


def df_columns_mapping(df):
    '''
    given a dataframe df
    get a dictionary mapping of col names to their list of values
    '''
    return {col: df[col] for col in df.columns.to_list()}


def create_lambda_function(formula):
    '''
    formula: string representation of a formula
    return a lambda function that takes in a dictionary mapping
    the variable names to a list of their column values
    '''

    def lambda_function(**variables):
        '''
        variables: dictionary mapping var name to
        a list of their values
        '''
        return eval(formula, {}, variables)

    return lambda_function


def modify_trl(df):
    # Remove rows where all TRL values are NaN
    df = df.dropna(subset=['START_TRL', 'CURRENT_TRL', 'END_TRL'], how='all')

    def fill_trl_values(row):
        # If CURRENT_TRL is NaN and STATUS_ENCODED is 1, use END_TRL or START_TRL if available
        if pd.isna(row['CURRENT_TRL']) and row['STATUS_ENCODED'] == 1:
            row['CURRENT_TRL'] = row['END_TRL'] if pd.notna(
                row['END_TRL']) else row['START_TRL']

        # If CURRENT_TRL is still NaN but both START_TRL and END_TRL exist, take the midpoint
        if pd.isna(row['CURRENT_TRL']) and pd.notna(row['START_TRL']) and pd.notna(row['END_TRL']):
            row['CURRENT_TRL'] = (row['START_TRL'] + row['END_TRL']) // 2

        # If END_TRL is only non-NaN, send START_TRL AND CURRENT_TRL to END_TRL
        if pd.isna(row['CURRENT_TRL']) and pd.isna(row['START_TRL']) and pd.notna(row['END_TRL']):
            row['START_TRL'] = row['END_TRL']
            row['CURRENT_TRL'] = row['END_TRL']

            # Assign default ranges for missing START_TRL and END_TRL based on CURRENT_TRL
        trl_ranges = [(1, 3), (4, 6), (7, 9)]
        for start, end in trl_ranges:
            if start <= row['CURRENT_TRL'] <= end:
                row['START_TRL'] = row['START_TRL'] if pd.notna(
                    row['START_TRL']) else start
                row['END_TRL'] = row['END_TRL'] if pd.notna(
                    row['END_TRL']) else end

        return row

    return df.apply(fill_trl_values, axis=1)


def clean_titles_column(df):
    """
    Shortens titles in the dataframe and removes phase information.
    """
    df['PROJECT_TITLE_CLEANED'] = df['PROJECT_TITLE'].apply(
        lambda title: re.sub(r'[^a-zA-Z]', '',  # Remove non-alphanumeric characters
                             # Remove "Phase I, II, etc."
                             re.sub(r',? Phase [I]+', '',
                                    str(title), flags=re.IGNORECASE)
                             ).lower()
    )
    return df
