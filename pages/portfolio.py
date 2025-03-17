import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from st_files_connection import FilesConnection
from src.data_collection import TechportData, SBIRData
from src.mpt_calc import get_mpt_investments
from utils.mpt_utils import df_columns_mapping, create_lambda_function

st.set_page_config(layout="wide",
                   initial_sidebar_state="expanded", page_icon='🛰️')

st.title("Portfolio Theory")
st.write(
    """
    The use of Mean-Variance Portfolio Optimization (MVO) in Modern Portfolio Theory (MPT) has been a long-standing method to guide investment decisions for market-traded assets like stocks and bonds. However, MPT lacks the ability to adapt strategies based on evolving market conditions. Experts using MPT do adapt portfolios, but such adaptation is carried out with expert judgement, and various approaches of predictive models and expectations of future returns. The fundamental MPT approach does not inherently include adaptive terms in its formulation. On the other hand, deep reinforcement learning (DRL) offers the following two advantages: capture time-varying market dynamics and accommodate non-linear patterns.
    """
)
with st.spinner("Loading modules...", show_time=True):
    with st.sidebar:

        # make connection to s3 bucket
        conn = st.connection('s3', type=FilesConnection)

        # get processed data to use as dataframe
        df = pd.merge(TechportData(conn).load_processed_data(),
                      SBIRData(conn).load_processed_data(),
                      on=['PROJECT_TITLE', 'START_YEAR', 'END_YEAR'], how='left')
        ### Dataset Info ###
        st.header('Dataset Information')
        if st.button("View Dataset"):
            st.switch_page('pages/data.py')

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
            default = "10 ** 5 * (0.4 * (CURRENT_TRL - START_TRL) / (END_TRL - START_TRL) + 0.3 * (CURRENT_TRL - START_TRL) / NUMBER_EMPLOYEES + 0.2 * LOG_VIEW_COUNT)"
            formula_input = st.text_input(
                "Custom Formula Using Numerical Variables Above (PERFORMANCE)", value=default)
            df['PERFORMANCE'] = create_lambda_function(
                formula_input)(**df_columns_mapping(df))

#######################################################
    # MPT
    with st.expander('[MPT]', expanded=True):
        st.header("Modern Portfolio Theory")
        st.write(
            """
            Pioneered by Harry Markowitz in the 1950s, mean-variance optimization (MVO) is a quantitative framework that allocates a budget to construct an optimal portfolio by maximizing returns for a given level of risk (or minimize risk for a given level of return). Risk is measured by the volatility of each asset and return is estimated by leveraging historical pricing data of the asset. With this risk and return setup, we have the following quadratic optimization problem:
            """
        )
        st.latex(r'''
                \begin{align*}
            \text{ maximize   } &\\mu^T w - \lambda w^T \Sigma w \\
            \text{ subject to   } &w_i \\ge 0 \\
            & \sum w_i = 1
        \end{align*}
                ''')
        st.write(
            """
            where $w$ is the weight vector for a set of assets where $w_i$ is the proportion of the budget invested in asset $i$, $\mu$ is the vector for the expected returns where $\mu_i$ is the expected return of asset $i$, and $\Sigma$ is the covariance matrix that represents the relationship between the returns of the assets.
            ***
            """
        )
        mpt_investments, portfolio_returns, portfolio_risks = get_mpt_investments(
            df)

        # Scatter Plot
        scatter_pl_data = {
            "Portfolio": [i for i in range(1, 11)],
            "Portfolio Risk": [round(risk, 2) for risk in portfolio_risks],
            "Portfolio Mean Return": [round(ret, 2) for ret in portfolio_returns]
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

        # Pie Charts
        df_pie = pd.DataFrame({
            "Cluster": [f"Cluster #{i}" for i in range(1, 8)],
            **{f"Portfolio #{i+1}": values for i, values in enumerate(mpt_investments)}
        })

        # Define subplot layout
        fig_pie = make_subplots(
            rows=5, cols=2,
            specs=[[{'type': 'domain'}] * 2] * 5,
            subplot_titles=[f"Portfolio #{i+1}" for i in range(10)]
        )

        # Add pie charts to subplots
        for i, (row, col) in enumerate([(r, c) for r in range(1, 6) for c in range(1, 3)]):
            fig_pie.add_trace(
                go.Pie(
                    values=df_pie[f"Portfolio #{i+1}"],
                    labels=df_pie["Cluster"],
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

        # Proportional Stacked Bar Chart
        stack_bar_ch_data = {
            "Portfolio": [f"Portfolio #{i}" for i in range(1, 11)],
            **{f"Cluster #{i + 1}": values for i, values in enumerate(mpt_investments.T)}
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
            text=df_bar_melted["Percentage"].apply(lambda x: f"{x:.1f}%")
        )

        fig_bar.update_traces(
            textposition="inside",
            hovertemplate="<b>%{x}</b><br>Percentage: %{y:.1f}%"
        )

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
    with st.expander('[DRL]', expanded=True):
        st.header("Deep Reinforcement Learning")
        st.write(
            """
            Reinforcement learning is a machine learning technique that learns how to make sequential decisions in an environment by optimizing long-term rewards. We formalize the environment with a Markov Decision Process (MDP). An MDP consists of 5 elements $(S, A, T, R, \gamma)$:
            - $S$ is the state space.
            - $A$ is the action space.
            - $T: S \\times A \\times S \\rightarrow R$ is the *transition model* where $T(s, a, s') = \\text{Pr}(s_{t+1}=s' | s_t = s, a_t=a)$ is the probability that given being in state $s \in S$ at time $t$ and taking action $a \in A$ will lead to $s'$ at time $t+1$.
            - $R: S \\times A \\rightarrow R$ is the *reward function* where $R(s, a)$ is the immediate reward for taken action $a$ when in state $s$.
            - $\gamma \in [0,1]$ is the hyperparameter *discount factor* that represents the balance between short-term gains and long-term rewards for the decision-making process.

            The solution to an MDP is a policy $\pi:S\\rightarrow A$ that specifies what action to take at every state. In large and continuous state spaces such as the economic market, the field of Deep Reinforcement Learning (DRL) utilizes neural networks as function approximators to learn estimate state-action value functions and learn $\pi$. As a result, the transition function $T$ is not explicitly modeled.
            ***
            """
        )
