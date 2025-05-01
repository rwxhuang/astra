import pandas as pd
import numpy as np


def create_balanced_timeseries(df):
    df = df[['UTILITY', 'CLUSTER', 'START_DATE']].copy()
    df['START_DATE'] = pd.to_datetime(df['START_DATE'])
    df = df.sort_values(['CLUSTER', 'START_DATE'])
    grouped = df.groupby('CLUSTER')

    # Step 5: Find the minimum number of entries across groups
    min_count = grouped.size().min()

    # Step 6: Sample min_count from each group while keeping the START_DATE order
    balanced_df = grouped.head(min_count).reset_index(drop=True)

    # Step 7: Sort the combined DataFrame by START_DATE to align by temporal index
    balanced_df = balanced_df.sort_values('START_DATE').reset_index(drop=True)

    # Step 8: Assign a time index
    balanced_df['TIME_INDEX'] = balanced_df.groupby(
        'CLUSTER').cumcount() + 1

    # Step 9: Pivot the df
    pivoted_df = balanced_df.pivot(
        index='TIME_INDEX', columns='CLUSTER', values='UTILITY')

    return pivoted_df


def get_drl_investments(df):
    time_df = create_balanced_timeseries(df)

    return np.array([[0.2, 0.3, 0.4, 0.1]])
