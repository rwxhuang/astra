import streamlit as st
import pandas as pd

from st_files_connection import FilesConnection
from src.data_collection import TechportData
from utils.data_utils import df_columns_mapping, create_lambda_function

st.set_page_config(
    layout="wide", initial_sidebar_state="expanded", page_icon='🛰️')

st.header("Markowitz Portfolio Theory")

with st.sidebar:
    
    # make connection to s3 bucket
    conn = st.connection('s3', type=FilesConnection)

    # get processed data to use as dataframe
    techport_data = TechportData(conn)
    techport_df = techport_data.load_processed_data()


    ### Dataset Info ###
    st.header('Dataset Information')

    ### numerical variables list ###
    st.text("Available Numerical Variables")
    numerical_vars_list = st.sidebar.empty()

    ### custom utility function ###

    # default util function
    default = "START_TRL / END_TRL"
    formula_input = st.text_input("Custom Formula Using Numerical Variables Above (UTILITY)", value=default)

    # custom function code
    techport_df_columns = df_columns_mapping(techport_df)
    custom_function = create_lambda_function(formula_input)
    techport_df['UTILITY'] = custom_function(**techport_df_columns)

    # add to empty vars list initialized above
    numerical_cols = techport_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_vars_list.code("\n".join(numerical_cols))

    
    ### Projects to Invest In ###
    st.header('Projects to Invest In')

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
        projects_df = techport_df.head(10)
    
    st.dataframe(projects_df)


    ### optional custom modifications ###

    with st.expander('Optional custom modifications', expanded=False):
        st.slider('slider', min_value=0., max_value=1., step=0.01)
        
        # make new df using inputted project ids or default ones
        min_max_df = pd.DataFrame({  
                        'min_frac': 0,  
                        'max_frac': 1   
                        },
                        index=projects_df.index
                        )
        
        st.dataframe(min_max_df)
