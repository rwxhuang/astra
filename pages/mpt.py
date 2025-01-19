import streamlit as st
import pandas as pd
# from utils.data_utils import create_lambda_function, df_columns_mapping

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

st.set_page_config(
    layout="wide", initial_sidebar_state="expanded", page_icon='🛰️')

st.header("Markowitz Portfolio Theory")

with st.sidebar:
    st.header('Data Upload')
    uploaded_data_file = st.file_uploader(
        'Upload technology project data'
    )


    ### numerical variables portion ###


    st.text("Available Numerical Variables")

    numerical_cols = []

    # process the uploaded file
    if uploaded_data_file:

        techport_df = pd.read_csv(uploaded_data_file)
        
        # get int and float cols
        numerical_cols = techport_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        ### custom utility function ###
    
    # display the numerical variables
    st.code("\n".join(numerical_cols))

    formula_input = st.text_input("Custom Formula Using Numerical Variables Above", value="")

    if formula_input:

        custom_function = create_lambda_function(formula_input)

        techport_df_columns = df_columns_mapping(techport_df)

        techport_df['UTILITY'] = custom_function(**techport_df_columns)



    
    st.header('Projects to Invest In')
    upload_project_file = st.file_uploader(
        'Upload technology projects to invest in'
    )

    st.write("Tech Projects:")

    if upload_project_file:

        project_df = pd.read_csv(upload_project_file)
        st.dataframe(project_df)


    with st.expander('Optional custom modifications', expanded=False):
        st.slider('slider', min_value=0., max_value=1., step=0.01)

        if upload_project_file:

            custom_mod_df = pd.DataFrame({
                            'Project ID': project_df['PROJECT_ID'],  
                            'min_frac': 0,  
                            'max_frac': 1   
                            })
            st.dataframe(custom_mod_df)
