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
        one_hot_df = pd.DataFrame(df[col].tolist(), index=df.index, columns=[
                                  f"{col}_{i}" for i in range(len(df[col].iloc[0]))])
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


def get_mu_cov_of_clusters(df, use_kmeans, cols, num_clusters):
    '''
    df: dataframe merged with techport and SBIR data
    returns a 10 x num_clusters matrix 
    '''

    # assign each project a cluster grouping
    if use_kmeans:
        get_kmeans_cluster(df, cols,
                           [
                               "START_DATE",
                               "END_DATE",
                               "LAST_MODIFIED"
                           ],
                           [
                               'TX_LEVEL_ENCODED_SUBLEVEL',
                               'LOCATIONS_ENCODED',
                           ],
                           num_clusters
                           )
    else:
        df['CLUSTER'] = df[cols[0]]

    # replace the missing award amounts with the mean of each cluster
    # get the average award amounts of each cluster
    average_cluster_award = df.groupby(
        'CLUSTER')['AWARD_AMOUNT'].mean().to_dict()

    # replace nan's in award amount
    df['AWARD_AMOUNT'] = df.apply(lambda row: average_cluster_award[row['CLUSTER']] if pd.isna(
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
    cluster_names = cluster_utilities.columns.to_list()
    mu = np.array(cluster_utilities.mean())
    cov = np.matrix(cluster_utilities.cov())

    return df, cluster_names, mu, cov


def get_mpt_investments(df, use_kmeans, cols, num_clusters):

    df, cluster_names, mu, cov = get_mu_cov_of_clusters(
        df, use_kmeans, cols, num_clusters)

    # calculate min return
    ef_min = EfficientFrontier(mu, cov)
    min_return = np.dot(np.array(list(ef_min.min_volatility().values())), mu)

    # calculate max return
    ef_max = EfficientFrontier(mu, cov)
    max_return = ef_max._max_return()

    # calculate portfolios
    ef_portfolio = EfficientFrontier(mu, cov)
    num_options = 10
    portfolio_returns = np.linspace(
        min_return, max_return, num=num_options + 1)[:-1]
    weights = [
        ef_portfolio.efficient_return(portfolio_return)
        for portfolio_return in portfolio_returns
    ]
    # make a 10 x # of clusters matrix
    weights_matrix = np.array([list(weights.values()) for weights in weights])

    # calculate risk
    portfolio_risks = [sum([weights_matrix[i][j] ** 2 * mu[j]
                            for j in range(len(weights_matrix[i]))]) ** 0.5 for i in range(num_options)]

    return df, cluster_names, weights_matrix, portfolio_returns, portfolio_risks


def merge_dfs(techport_df, sbir_df):
    """
    Merges Techport and SBIR dataframes based on unique project titles and start years,
    while ensuring duplicated columns are not included in the final result.
    """
    def get_unique_matches(df1, df2, keys):
        counts1, counts2 = df1[keys].value_counts(), df2[keys].value_counts()
        unique_keys = counts1[counts1 == 1].index.intersection(
            counts2[counts2 == 1].index)
        return df1[df1.set_index(keys).index.isin(unique_keys)], df2[df2.set_index(keys).index.isin(unique_keys)]

    def merge_without_duplicates(df1, df2, on_keys, how='left', validate=None):
        common_cols = set(df1.columns).intersection(df2.columns) - set(on_keys)
        df2 = df2.drop(columns=common_cols, errors='ignore')
        return pd.merge(df1, df2, on=on_keys, how=how, validate=validate)

    # Step 1: Match on 'PROJECT_TITLE_CLEANED'
    techport_unique, sbir_unique = get_unique_matches(
        techport_df, sbir_df, ['PROJECT_TITLE_CLEANED'])
    df_unique_merged = merge_without_duplicates(
        techport_unique, sbir_unique, ['PROJECT_TITLE_CLEANED'], validate='one_to_one')

    # Step 2: Match on 'PROJECT_TITLE_CLEANED' and 'START_YEAR'
    remaining_techport = techport_df[~techport_df.index.isin(
        techport_unique.index)]
    remaining_sbir = sbir_df[~sbir_df.index.isin(sbir_unique.index)]
    techport_unique2, sbir_unique2 = get_unique_matches(
        remaining_techport, remaining_sbir, ['PROJECT_TITLE_CLEANED', 'START_YEAR'])
    df_unique_merged2 = merge_without_duplicates(
        techport_unique2, sbir_unique2, ['PROJECT_TITLE_CLEANED', 'START_YEAR'], validate='one_to_one')

    # Step 3: Merge remaining data including 'END_YEAR'
    remaining_techport2 = remaining_techport[~remaining_techport.index.isin(
        techport_unique2.index)]
    remaining_sbir2 = remaining_sbir[~remaining_sbir.index.isin(
        sbir_unique2.index)]
    df_remaining_merged2 = merge_without_duplicates(
        remaining_techport2, remaining_sbir2, ['PROJECT_TITLE_CLEANED', 'START_YEAR', 'END_YEAR'])

    merged_df = pd.concat(
        [df_unique_merged, df_unique_merged2, df_remaining_merged2], ignore_index=True)

    return merged_df
