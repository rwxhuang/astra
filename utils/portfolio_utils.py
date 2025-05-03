import pandas as pd
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from st_files_connection import FilesConnection
from src.data_collection import TechportData, SBIRData
from src.mpt_calc import merge_dfs

CASE_STUDIES_IDX = {
    'Case Study 0: Live Techport Dataset': 0,
    'Case Study 1: TX 1,6,8,12': 1,
    'Case Study 2: Propulsion Systems (TX01.1-01.4)': 2,
    'Case Study 3: Aerospace Power and Energy Storage (TX03.1-03.3)': 3,
    'Case Study 4: Sensors & Instruments (TX08.1-08.3)': 4,
    'Case Study 5: Software, Modeling, Simulation, and Information Processing (TX11.1-11.6)': 5
}

CASE_STUDIES_QUERYS = [
    '',
    'TX_EXTRACTED_INT in [1, 6, 8, 12]',
    'TX_EXTRACTED in [1.1,1.2,1.3,1.4]',
    'TX_EXTRACTED in [3.1,3.2,3.3]',
    'TX_EXTRACTED in [8.1,8.2,8.3]',
    'TX_EXTRACTED in [11.1,11.2,11.3,11.4,11.5,11.6]',
]

CASE_STUDIES_VALUES = [
    True,
    False,
    False,
    False,
    False,
    False,
]

CASE_STUDIES_CLUSTER_COLS_AUTO = [
    [
        'START_TRL',
        'END_TRL',
        'CURRENT_TRL',
        'START_DATE',
        'VIEW_COUNT_NORMALIZED',
        'NUMBER_EMPLOYEES_NORMALIZED',
        'END_DATE',
        'TX_LEVEL_ENCODED_SUBLEVEL',
        'LOCATIONS_ENCODED',
        'STATUS_ENCODED',
        'LAST_MODIFIED'
    ] for _ in range(6)
]
CASE_STUDIES_CLUSTER_COLS_MANUAL = [
    ['TX_EXTRACTED_INT'],
    ['TX_EXTRACTED_INT'],
    ['TX_EXTRACTED'],
    ['TX_EXTRACTED'],
    ['TX_EXTRACTED'],
    ['TX_EXTRACTED'],
]

CASE_STUDIES_NUM_CLUSTERS = [7, 4, 4, 3, 3, 6]


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


def get_df(id):
    # @st.cache_data(show_spinner=False)
    def fetch_df_from_s3():
        conn = st.connection('s3', type=FilesConnection)
        return merge_dfs(
            TechportData().load_processed_data(),
            SBIRData(conn).load_processed_data()
        )
    df = fetch_df_from_s3()

    if id:
        return df.query(CASE_STUDIES_QUERYS[id])
    return df


def get_scatter_plot(investments, portfolio_returns, portfolio_risks):
    scatter_pl_data = {
        "Portfolio": [i for i in range(1, investments.shape[0] + 1)],
        "Portfolio Risk": [risk for risk in portfolio_risks],
        "Portfolio Mean Return": [ret for ret in portfolio_returns]
    }

    df_scatter = pd.DataFrame(scatter_pl_data)
    df_scatter["Custom Label"] = [
        f"Portfolio #{i}" for i in range(1, len(df_scatter) + 1)]

    # Create the scatter plot
    fig_scatter = px.line(
        df_scatter,
        markers=True,
        x="Portfolio Risk",
        y="Portfolio Mean Return",
        text="Custom Label",
        labels={"Portfolio Risk": "Risk of Portfolio (std dev.)",
                "Portfolio Mean Return": "Mean Return of Portfolios"},
        template="plotly_dark"
    )
    fig_scatter.update_traces(
        marker=dict(size=9),
        hovertemplate="<b>%{text}</b><br>Risk: %{x:.2f}<br>Return: %{y:.2f}"
    )
    fig_scatter.update_layout(
        xaxis=dict(showgrid=True),  # Show grid on x-axis
        yaxis=dict(showgrid=True)   # Show grid on y-axis
    )

    return fig_scatter


def get_pie_charts(investments, labels, color_map):
    df_pie = pd.DataFrame({
        'LABEL': labels,
        **{f"Portfolio #{i+1}": values for i, values in enumerate(investments)}
    })

    # Define subplot layout
    fig_pie = make_subplots(
        rows=5, cols=2,
        specs=[[{'type': 'domain'}] * 2] * 5,
        subplot_titles=[
            f"Portfolio #{i+1}" for i in range(investments.shape[0])
        ]
    )

    # Add sorted pie charts to subplots
    for i, (row, col) in enumerate([(r, c) for r in range(1, 6) for c in range(1, 3)]):
        # Add pie chart to subplot
        fig_pie.add_trace(
            go.Pie(
                values=df_pie[f"Portfolio #{i+1}"].values,
                labels=df_pie['LABEL'].values,
                marker=dict(colors=[color_map[label]
                            for label in df_pie['LABEL'].values]),
                hovertemplate="<b>%{label}</b><br>Percentage: %{percent:.2f}",
                # Show legend only for the first pie chart
                showlegend=(i == 0)
            ),
            row=row, col=col
        )

    # Configure layout
    fig_pie.update_layout(
        height=1000,
        width=800,
        font_size=12,
        legend=dict(
            orientation="h",
            y=1.17,
            x=0.25,
            xanchor="center",
            traceorder="normal"
        )
    )

    return fig_pie


def get_bar_chart(investments, labels, color_map):
    stack_bar_ch_data = {
        "Portfolio": [f"Portfolio #{i}" for i in range(1, investments.shape[0] + 1)],
        **{labels[i]: values for i, values in enumerate(investments.T)}
    }

    df_stack_bar = pd.DataFrame(stack_bar_ch_data)

    # Normalize values to percentages
    df_bar_normalized = df_stack_bar.set_index("Portfolio")
    df_bar_normalized = df_bar_normalized.div(
        df_bar_normalized.sum(axis=1), axis=0) * 100
    df_bar_normalized.reset_index(inplace=True)

    # Melt data for Plotly
    df_bar_melted = df_bar_normalized.melt(
        id_vars="Portfolio", var_name="Clusters", value_name="Percentage"
    )

    # Create stacked bar chart with formatted percentages
    fig_bar = px.bar(
        df_bar_melted,
        x="Portfolio",
        y="Percentage",
        color="Clusters",
        barmode="stack",
        text=df_bar_melted["Percentage"].apply(lambda x: f"{x:.1f}%"),
        color_discrete_map=color_map
    )

    fig_bar.update_traces(
        textposition="inside",
        hovertemplate="<b>%{x}</b><br>Percentage: %{y:.1f}%"
    )
    return fig_bar


def get_drl_pie(investments, labels):
    # Create a DataFrame for the pie chart
    df_pie = pd.DataFrame({
        'LABEL': labels,
        **{f"Portfolio #{i+1}": values for i, values in enumerate(investments)}
    })

    # Create the pie chart
    fig_pie = px.pie(
        df_pie,
        values=df_pie.iloc[:, 1:].sum(axis=1),
        names=df_pie['LABEL'],
        color_discrete_sequence=px.colors.qualitative.Plotly,
        template="plotly_dark"
    )

    fig_pie.update_layout(
        title="DRL Investment Decision",
        title_font_size=24,
        font=dict(size=14)  # Applies to legend, labels, etc.
    )

    return fig_pie
