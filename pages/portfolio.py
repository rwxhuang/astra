import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.colors

from src.mpt_calc import get_mpt_investments
from src.drl_calc import get_drl_investments
from utils.portfolio_utils import *

st.set_page_config(layout="wide",
                   initial_sidebar_state="expanded", page_icon='🛰️')

st.title("Portfolio Theory")
st.write(
    """
    The use of Mean-Variance Portfolio Optimization (MVO) in Modern Portfolio Theory (MPT) has been a long-standing method to guide investment decisions for market-traded assets like stocks and bonds. However, MPT lacks the ability to adapt strategies based on evolving market conditions. Experts using MPT do adapt portfolios, but such adaptation is carried out with expert judgement, and various approaches of predictive models and expectations of future returns. The fundamental MPT approach does not inherently include adaptive terms in its formulation. On the other hand, deep reinforcement learning (DRL) offers the following two advantages: capture time-varying market dynamics and accommodate non-linear patterns.
    """
)
with st.spinner("*Loading modules...*", show_time=True):
    with st.sidebar:
        ### STEP 1. Load Data ###
        st.markdown("**Step 1. Load Technology Projects Data**",
                    help='Insert Information here')
        case_study_idx = CASE_STUDIES_IDX[st.selectbox(
            'Select a case study:', CASE_STUDIES_IDX.keys(), index=0)]
        with st.spinner(''):
            df = get_df(case_study_idx)
        ### STEP 2. K-Means Clustering vs Manual clustering ###
        st.markdown("**Step 2. Create Clusters**",
                    help='Insert Information here')
        _, cluster_col, _, = st.columns(3)
        use_kmeans = st.toggle(
            'Automatic', value=CASE_STUDIES_VALUES[case_study_idx])
        # Display columns selection for clustering
        if use_kmeans:
            num_clusters = st.slider(
                'Number of Clusters', min_value=1, max_value=10, step=1, value=CASE_STUDIES_NUM_CLUSTERS[case_study_idx])
            cols = st.multiselect(
                'Columns for clustering',
                df.columns,
                default=CASE_STUDIES_CLUSTER_COLS_AUTO[case_study_idx]
            )
        else:
            cols = [
                st.selectbox(
                    'Cluster Column',
                    df.columns,
                    index=df.columns.get_loc(
                        CASE_STUDIES_CLUSTER_COLS_MANUAL[case_study_idx][0])
                )
            ]
        ### STEP 3. Set Performance Metric  ###
        st.markdown("**Step 3. Set Performance Metric**",
                    help='Insert Information here')
        st.caption("Available Numerical Variables", unsafe_allow_html=True)
        numerical_cols = df.select_dtypes(
            include=['int64', 'float64']).columns.tolist()
        numerical_cols = [
            col for col in numerical_cols if "unnamed" not in col.lower()]
        st.code("\n".join(numerical_cols))
        formula_input = st.text_input(
            "Custom Formula Using Numerical Variables Above (PERFORMANCE)",
            value="10 ** 5 * (4/9 * (CURRENT_TRL - START_TRL + 0.5) / (END_TRL - START_TRL + 0.5) + 3/9 * (CURRENT_TRL - START_TRL + 0.5) / (END_YEAR - START_YEAR + 0.5) + 2/9 * (CURRENT_TRL - START_TRL + 0.5) / (NUMBER_EMPLOYEES+0.5) + 0 * LOG_VIEW_COUNT)"
        )
        df['NUMBER_EMPLOYEES'] = df['NUMBER_EMPLOYEES'].fillna(
            df['NUMBER_EMPLOYEES'].mean())
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
            \text{ maximize   } &\mu^T w - \lambda w^T \Sigma w \\
            \text{ subject to   } &w_i \ge 0 \\
            & \sum w_i = 1
        \end{align*}
                ''')
        st.write(
            """
            where $w$ is the weight vector for a set of assets where $w_i$ is the proportion of the budget invested in asset $i$, $\mu$ is the vector for the expected returns where $\mu_i$ is the expected return of asset $i$, and $\Sigma$ is the covariance matrix that represents the relationship between the returns of the assets.
            ***
            """
        )
        # MPT Calculations
        df, cluster_names, mpt_investments, portfolio_returns, portfolio_risks = get_mpt_investments(
            df,
            use_kmeans,
            cols,
            num_clusters if use_kmeans else None
        )
        # Color Configurations
        labels = [
            f"Cluster #{i}" if use_kmeans else f"{cols[0]} {cluster_names[i - 1]}"
            for i in range(1, mpt_investments.shape[1] + 1)
        ]
        color_palette = plotly.colors.qualitative.Plotly
        unique_labels = labels
        color_map = {label: color_palette[i % len(
            color_palette)] for i, label in enumerate(unique_labels)}
        # Scatter Plot
        fig_scatter = get_scatter_plot(
            mpt_investments,
            portfolio_returns,
            portfolio_risks
        )
        # Pie Charts
        fig_pie = get_pie_charts(mpt_investments, labels, color_map)
        # Proportional Stacked Bar Chart
        fig_bar = get_bar_chart(mpt_investments, labels, color_map)
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
        # drl_investments = get_drl_investments(df)
        # fig_drl_pie = get_drl_pie(drl_investments, labels)

        # st.plotly_chart(fig_drl_pie, use_container_width=True)
        st.warning('Coming soon: to be implemented', icon="⚠️")
