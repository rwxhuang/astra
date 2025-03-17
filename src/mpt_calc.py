import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from pypfopt.efficient_frontier import EfficientFrontier


def get_kmeans_cluster(df, cluster_cols, date_cols, encoded_cluster_cols, num_clusters):
    '''
    df: dataframe of merged techport and SBIR data
    cluster_cols: cols to apply kmeans clustering on as a list of string names
    encoded_cluster_cols: which cols that are one-hot encodings to apply kmeans clustering on as a list of string names
    num_clusters: number of clusters to group into

    applies a kmeans clustering on the given df against the given cols and returns a
    frequency map of the cluster number and the number of projects it contains

    cleaning data: 
    normalize datetime values to a range from 0 to 1
    replace nan's with medians of cols
    flatten one-hot encodings into their own cols
    '''

    # normalize dates to a float
    scaler = MinMaxScaler(feature_range=(0, 1))

    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
        df[col] = df[col].apply(lambda x: x.timestamp())
        df[col] = scaler.fit_transform(df[[col]])

    # get cols to add as dimensions for clustering
    cluster_df = df[cluster_cols]

    for col in encoded_cluster_cols:
        # apply conversion from string to list on all cols
        one_hot_data = cluster_df[col].apply(list)
        # for each entry in list, flatten into its own col
        one_hot_df = pd.DataFrame(one_hot_data.tolist(), columns=[
                                  f"{col}_{i}" for i in range(one_hot_data.iloc[0].__len__())])

        # add to original clustering df
        cluster_df = pd.concat([cluster_df, one_hot_df], axis=1)

    # drop the original one-hot encoded columns if no longer needed
    cluster_df = cluster_df.drop(columns=encoded_cluster_cols)

    cols_with_nans = cluster_df.columns[cluster_df.isna().any()].tolist()

    # replace nan's with the median value of the col
    for col in cols_with_nans:
        cluster_df[col] = cluster_df[col].fillna(cluster_df[col].median())

    # begin kmeansclustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(cluster_df)

    # add cluster to original df with all cols
    df['CLUSTER'] = kmeans.labels_

    # make a dictionary mapping each
    return df


def get_mu_cov_of_clusters(df):
    '''
    df: dataframe merged with techport and SBIR data
    returns a 10 x 7 matrix 
    '''

    # assign each project a cluster grouping
    df = get_kmeans_cluster(df,
                            [
                                'START_TRL',
                                'END_TRL',
                                'CURRENT_TRL',
                                'START_DATE',
                                'VIEW_COUNT_NORMALIZED',
                                'NUMBER_EMPLOYEES_NORMALIZED',
                                'END_DATE',
                                'TX_LEVEL_ENCODED',
                                'LOCATIONS_ENCODED',
                                'STATUS_ENCODED',
                                'LAST_MODIFIED'
                            ],
                            [
                                "START_DATE",
                                "END_DATE",
                                "LAST_MODIFIED"
                            ],
                            [
                                'TX_LEVEL_ENCODED',
                                'LOCATIONS_ENCODED',
                                'STATUS_ENCODED'
                            ],
                            7
                            )

    # replace the missing award amounts with the mean of each cluster
    # get the average award amounts of each cluster
    average_cluster_value = df.groupby(
        'CLUSTER')['AWARD_AMOUNT'].mean().to_dict()

    # replace nan's in award amount
    df['AWARD_AMOUNT'] = df.apply(lambda row: average_cluster_value[row['CLUSTER']] if pd.isna(
        row['AWARD_AMOUNT']) else row['AWARD_AMOUNT'], axis=1)

    # df['PERFORMANCE'] = df.apply(calculate_performance, axis=1)
    df['UTILITY'] = df['PERFORMANCE'] / df['AWARD_AMOUNT']

    # make same number of projects in each cluster
    min_count = df.groupby('CLUSTER').size().min()
    reduced_df = df.groupby('CLUSTER').apply(
        lambda x: x.sample(n=min_count)).reset_index(drop=True)

    # make a df of cluster groups as cols and utility values as row values
    utility_cluster_df = reduced_df[['UTILITY', 'CLUSTER']]
    cluster_utilities_as_list = utility_cluster_df.groupby('CLUSTER')[
        'UTILITY'].apply(list)
    cluster_utilities = pd.DataFrame(dict(cluster_utilities_as_list))

    # get mean of clusters and covariance matrix
    mu = cluster_utilities.mean()
    cov = cluster_utilities.cov()

    return df, mu, cov


def get_mpt_investments(df):

    _, mu, cov = get_mu_cov_of_clusters(df)
    print(mu, cov)

    # calculate min return
    ef_min = EfficientFrontier(mu, cov)
    min_return = np.dot(np.array(list(ef_min.min_volatility().values())), mu)

    # calculate max return
    ef_max = EfficientFrontier(mu, cov)
    max_return = ef_max._max_return()

    # calculate portfolios
    ef_portfolio = EfficientFrontier(mu, cov)
    num_options = 10
    portfolio_returns = np.linspace(min_return, max_return, num=num_options)

    weights = [ef_portfolio.efficient_return(
        portfolio) for portfolio in portfolio_returns]

    # make a 10 x # of clusters matrix
    weights_matrix = np.array([list(weights.values()) for weights in weights])

    # calculate risk
    portfolio_risks = [sum([weights_matrix[i][j] ** 2 * mu[j]
                            for j in range(len(weights_matrix[i]))]) ** 0.5 for i in range(num_options)]
    return weights_matrix, portfolio_returns, portfolio_risks
