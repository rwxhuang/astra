import streamlit as st
import pandas as pd

from st_files_connection import FilesConnection
from src.data_collection import TechportData, SBIRData
from src.mpt_calc import get_mpt_investments
from utils.mpt_utils import df_columns_mapping, create_lambda_function

st.set_page_config(
    layout="wide", initial_sidebar_state="expanded", page_icon='🛰️')

st.header("Markowitz Portfolio Theory")

with st.sidebar:

    # make connection to s3 bucket
    conn = st.connection('s3', type=FilesConnection)

    # get processed data to use as dataframe
    df = pd.merge(TechportData(conn).load_processed_data(),
                  SBIRData(conn).load_data(),
                  on=['PROJECT_TITLE', 'START_YEAR', 'END_YEAR'], how='left')

    ### Dataset Info ###
    st.header('Dataset Information')

    with st.container(border=True):
        ### numerical variables list ###
        st.text("Available Numerical Variables")

        # vars list
        numerical_cols = df.select_dtypes(
            include=['int64', 'float64']).columns.tolist()
        st.code("\n".join(numerical_cols))

        ### custom utility function ###

        # ui input section
        default = "START_TRL / END_TRL"
        formula_input = st.text_input(
            "Custom Formula Using Numerical Variables Above (UTILITY)", value=default)

        # custom function code
        techport_df_columns = df_columns_mapping(df)
        custom_function = create_lambda_function(formula_input)
        df['UTILITY'] = custom_function(**techport_df_columns)

    ### Projects to Invest In ###
    st.header('Projects to Invest In')

    with st.container(border=True):
        # user upload
        upload_project_file = st.file_uploader(
            'Upload technology projects to invest in'
        )

        ### show uploaded projects or default ###
        st.write("Tech Projects:")

        if upload_project_file:
            projects_df = pd.read_csv(upload_project_file)
        else:
            # default
            projects_df = df.head(10)

        st.dataframe(projects_df)

    ### optional custom modifications ###

    with st.expander('Optional custom modifications', expanded=False):

        # make new df using inputted project ids or default ones
        min_max_df = pd.DataFrame({
            'PROJECT_TITLE': projects_df['PROJECT_TITLE'],
            'min_frac': 0.0,
            'max_frac': 1.0
        },
            index=projects_df.index
        )

        # display the data as an editable table
        st.data_editor(min_max_df)

st.write('Investments:', get_mpt_investments(df))
