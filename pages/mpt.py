import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from st_files_connection import FilesConnection
from src.data_collection import TechportData, SBIRData
from src.mpt_calc import get_mpt_investments
from utils.mpt_utils import df_columns_mapping, create_lambda_function

st.header("Markowitz Portfolio Theory")

with st.sidebar:

    # make connection to s3 bucket
    conn = st.connection('s3', type=FilesConnection)

    # get processed data to use as dataframe
    df = pd.merge(TechportData(conn).load_processed_data(),
                  SBIRData(conn).load_processed_data(),
                  on=['PROJECT_TITLE', 'START_YEAR', 'END_YEAR'], how='left')

    ### Dataset Info ###
    st.header('Dataset Information')

    with st.container(border=True):
        ### numerical variables list ###
        st.text("Available Numerical Variables")

        # vars list
        numerical_cols = df.select_dtypes(
            include=['int64', 'float64']).columns.tolist()
        numerical_cols = [
            col for col in numerical_cols if "unnamed" not in col.lower()]
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

row_num = 5
col_num = 2
pie_chart_num = range(10)

# Create a subplot figure with domain type for pie charts
fig_pie = make_subplots(
    rows=row_num, cols=col_num,
    specs=[[{'type': 'domain'}] * col_num for _ in range(row_num)],
    subplot_titles=[f"Portfolio #{i+1}" for i in pie_chart_num]
)

# Function to create pie charts


def pie_chart(chart_num):
    return go.Pie(
        values=df_pie[f"Portfolio #{chart_num+1}"], labels=df_pie["Portfolio"],
        showlegend=(chart_num == 0)  # Show legend only for the first pie chart
    )


# Add pie charts to subplots
row, col = 1, 1
for i in pie_chart_num:
    fig_pie.add_trace(pie_chart(i), row=row, col=col)
    col += 1
    if col > col_num:  # Move to next row after filling columns
        col = 1
        row += 1

# Show legend
fig_pie.update_traces(showlegend=True)

# Update layout for better spacing
fig_pie.update_layout(
    height=1000,
    width=800,
    legend_title="Legend",
    font=dict(size=12),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5,
        traceorder='normal'
    )
)

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

fig_bar = px.bar(
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
    st.plotly_chart(fig_bar, use_container_width=True)

# Column 2: 10 Pie Charts in Two Subcolumns (5 per column)
with col2:
    st.subheader("Pie Charts for Calculated Portfolios")

    # Create two subcolumns inside column 2
    subcol1, subcol2 = st.columns(2)

    st.plotly_chart(fig_pie, use_container_width=True)
    # for i, fig in enumerate(pie_charts):
    #     if i < 5:  # First 5 pie charts go into subcol1
    #         with subcol1:
    #             st.plotly_chart(fig, use_container_width=False)
    #     else:  # Next 5 pie charts go into subcol2
    #         with subcol2:
    #             st.plotly_chart(fig, use_container_width=False)
