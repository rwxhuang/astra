import streamlit as st
import pandas as pd
import plotly.express as px

from st_files_connection import FilesConnection
from src.data_collection import TechportData
# from src.mpt_calc import get_mpt_investments
from utils.mpt_utils import df_columns_mapping, create_lambda_function

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

    with st.container(border=True):
        ### numerical variables list ###
        st.text("Available Numerical Variables")

        # vars list
        numerical_cols = techport_df.select_dtypes(
            include=['int64', 'float64']).columns.tolist()
        st.code("\n".join(numerical_cols))

        ### custom utility function ###

        # ui input section
        default = "START_TRL / END_TRL"
        formula_input = st.text_input(
            "Custom Formula Using Numerical Variables Above (UTILITY)", value=default)

        # custom function code
        techport_df_columns = df_columns_mapping(techport_df)
        custom_function = create_lambda_function(formula_input)
        techport_df['UTILITY'] = custom_function(**techport_df_columns)

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
            projects_df = techport_df.head(10)

        st.dataframe(projects_df)

    ### optional custom modifications ###

    with st.expander('Optional custom modifications', expanded=False):

        # make new df using inputted project ids or default ones
        min_max_df = pd.DataFrame({
            'min_frac': 0,
            'max_frac': 1
        },
            index=projects_df.index
        )

        st.dataframe(min_max_df)

#######################################################

# GRAPHS

# Scatter Plot
scatter_pl_data = {
    "Portfolio Risk": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Portfolio Mean Return": [10, 15, 7, 20, 25, 18, 30, 22, 12, 28]
}

df_scatter = pd.DataFrame(scatter_pl_data)

# Create the scatter plot
fig_scatter = px.scatter(
    df_scatter,
    x="Portfolio Risk",
    y="Portfolio Mean Return",
    labels={"Portfolio Risk": "Portfolio Risk (std dev.)",
            "Portfolio Mean Return": "Portfolio Mean Return"},
    template="plotly_dark"
)

# Pie Charts

pie_ch_data = {
    # "Portfolio #8", "Portfolio #9", "Portfolio #10"],
    "Portfolio": ["Portfolio #1", "Portfolio #2", "Portfolio #3", "Portfolio #4", "Portfolio #5", "Portfolio #6", "Portfolio #7"],
    "Portfolio #1": [20, 15, 10, 25, 5, 15, 10],
    "Portfolio #2": [30, 10, 20, 15, 5, 10, 10],
    "Portfolio #3": [25, 20, 15, 10, 10, 10, 10],
    "Portfolio #4": [10, 10, 10, 15, 20, 25, 10],
    "Portfolio #5": [15, 15, 15, 10, 10, 20, 15],
    "Portfolio #6": [5, 10, 20, 30, 10, 15, 10],
    "Portfolio #7": [10, 25, 20, 10, 10, 15, 10],
    "Portfolio #8": [20, 10, 30, 15, 10, 5, 10],
    "Portfolio #9": [15, 20, 10, 10, 15, 10, 20],
    "Portfolio #10": [10, 10, 15, 10, 20, 20, 15]
}

df_pie = pd.DataFrame(pie_ch_data)

pie_charts = []
for i in range(10):
    fig_pie = px.pie(df_pie, names="Portfolio", values=f"Portfolio #{
                     i+1}", title=f"Portfolio #{i+1}")
    fig_pie.update_layout(height=350, width=350)
    pie_charts.append(fig_pie)

# Proportional Stacked Bar Chart

stack_bar_ch_data = {
    "Portfolio": ["Portfolio #1", "Portfolio #2", "Portfolio #3", "Portfolio #4", "Portfolio #5", "Portfolio #6", "Portfolio #7", "Portfolio #8", "Portfolio #9", "Portfolio #10"],
    "Cluster #1": [20, 30, 25, 10, 15, 5, 10, 20, 15, 10],
    "Cluster #2": [15, 10, 20, 10, 15, 10, 25, 10, 20, 10],
    "Cluster #3": [10, 20, 15, 10, 15, 20, 20, 30, 10, 15],
    "Cluster #4": [25, 15, 10, 15, 10, 30, 10, 15, 10, 10],
    "Cluster #5": [5, 5, 10, 20, 10, 10, 10, 10, 15, 20],
    "Cluster #6": [15, 10, 10, 25, 20, 15, 15, 5, 10, 20],
    "Cluster #7": [10, 10, 10, 10, 15, 10, 10, 10, 20, 15]
}

df_stack_bar = pd.DataFrame(stack_bar_ch_data)

# normalize values
df_bar_normalized = df_stack_bar.set_index("Portfolio")
df_bar_normalized = df_bar_normalized.div(
    df_bar_normalized.sum(axis=1), axis=0) * 100
df_bar_normalized.reset_index(inplace=True)

# melt data for plotly
df_bar_melted = df_bar_normalized.melt(
    id_vars="Portfolio", var_name="Clusters", value_name="Percentage")

fig_bar_prop = px.bar(
    df_bar_melted,
    x="Portfolio",
    y="Percentage",
    color="Clusters",
    barmode="stack",
    text_auto=True
)

# Streamlit Layout

# Create two main columns
col1, col2 = st.columns([3, 2])

# Column 1: Proportional Stacked Bar Chart
with col1:
    st.subheader("Risk and Return for 10 Calculated Portfolios")
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Portfolio Weights for each Portfolio")
    st.plotly_chart(fig_bar_prop, use_container_width=True)

# Column 2: 10 Pie Charts in Two Subcolumns (5 per column)
with col2:
    st.subheader("Pie Charts for Calculated Portfolios")

    # Create two subcolumns inside column 2
    subcol1, subcol2 = st.columns(2)

    for i, fig in enumerate(pie_charts):
        if i < 5:  # First 5 pie charts go into subcol1
            with subcol1:
                st.plotly_chart(fig, use_container_width=False)
        else:  # Next 5 pie charts go into subcol2
            with subcol2:
                st.plotly_chart(fig, use_container_width=False)
