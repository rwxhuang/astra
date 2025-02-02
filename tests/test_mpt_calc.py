import pytest
import streamlit as st
import pandas as pd
import numpy as np

from st_files_connection import FilesConnection
from src.data_collection import TechportData, SBIRData
from src.mpt_calc import get_mpt_investments


@pytest.fixture
def dataframe():
    '''fixture to initialize dataframe'''
    conn = st.connection('s3', type=FilesConnection)
    return pd.merge(TechportData(conn).load_processed_data(),
                    SBIRData(conn).load_data(),
                    on=['PROJECT_TITLE', 'START_YEAR', 'END_YEAR'], how='left')


def test_mpt_investments_matrix(dataframe):
    """test matrix dimensions and rows sum to 1"""
    investment_matrix = get_mpt_investments(dataframe)
    assert investment_matrix.shape == (
        10, 7), "Investment matrix has incorrect dimensions"
    row_sums = investment_matrix.sum(axis=1)
    for i, sum in enumerate(row_sums):
        assert np.isclose(sum, 1, atol=1e-6), f"Row {i + 1} does not sum to 1"
    assert (investment_matrix >= 0).all().all(
    ), "Dataframe contains negative values"
