
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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
    dataframe['TX_PART2'] = dataframe['PRIMARY_TX'].str.extract(r'TX\d{2}\.(\d+|X)')

    # handles edge cases
    dataframe['TX_PART2'] = dataframe['TX_PART2'].replace('X', '0').fillna('0')

    # combine parts into a float for extracted tx
    dataframe['TX_EXTRACTED'] = dataframe['TX_PART1'] + '.' + dataframe['TX_PART2']
    dataframe['TX_EXTRACTED'] = dataframe['TX_EXTRACTED'].astype(float)

    # removes temp cols
    dataframe = dataframe.drop(columns=['TX_PART1', 'TX_PART2'])

    return dataframe

def encode_locations(dataframe):
    '''
    dataframe: given a dataframe
    returns a dataframe with col for locations encoded as a one-hot encoding
    '''
    # get all different combinations of locations listed
    locations_list = dataframe['LOCATIONS_WHERE_WORK_IS_PERFORMED'].unique()

    # initialize set to store unique locations
    unique_locations = []

    # for each combination
    for elem in locations_list:
        # if string
        if isinstance(elem, str):
            # try to split it 
            locations = elem.split('; ')
            # add each split location to set
            for location in locations:
                unique_locations.add(location)
        # handles NaN's (not strings)
        else:
            continue
    
    # not locations or repeated
    unique_locations.remove('Not Applicable')
    unique_locations.remove('Outside the United States ')

    # mapping from location to index
    location_to_index = {location: index for index, location in enumerate(unique_locations)}

    # repeat locations maps to same location in encoding
    location_to_index['Outside the United States '] = location_to_index['Outside the United States']


    def convert_to_vector_locations(locations):
        '''
        Takes in locations with form state; state; ...
        and converts it to a one-hot encoding [0,0,0,1,0...]
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

    # apply function created to all entries in col
    dataframe['LOCATIONS_ENCODED'] = dataframe['LOCATIONS_WHERE_WORK_IS_PERFORMED'].apply(convert_to_vector_locations)

    return dataframe

def encode_status(dataframe):
    '''
    dataframe: given a dataframe
    return the dataframe with a col for encoding for completed, active, or cancelled
    '''

    statuses = ['Active', 'Completed', 'Canceled']

    # mapping from status to index
    status_to_index = {status: index for index, status in enumerate(statuses)}

    def convert_to_vector_statuses(status):
        '''
        one hot encodes the given status
        '''
        # initialize empty binary vector
        vector = np.zeros(len(statuses), dtype=int)

        # change index corresponding to status to a 1
        vector[status_to_index[status]] = 1

        return vector
    
    # apply function created to all entries in col
    dataframe['STATUS_ENCODED'] = dataframe['STATUS'].apply(convert_to_vector_statuses)
    
    return dataframe

def normalize_views(dataframe, lower, upper):
    '''
    dataframe: takes in a dateframe
    lower: lower bound min
    upper: upper bound max
    returns a dataframe with the views normalized from lower to upper
    '''

    # initialize scaler object with lower and upper bounds
    scaler = MinMaxScaler(feature_range=(lower,upper))

    # apply scaler to col
    dataframe['VIEW_COUNT_NORMALIZED'] = scaler.fit_transform(dataframe[['VIEW_COUNT']])
    
    return dataframe
