import streamlit as st

st.set_page_config(layout="wide", page_icon='🛰️',
                   initial_sidebar_state=st.session_state.sidebar_state)

_, body, _ = st.columns([1, 6, 1])
with body:
    st.markdown("""
                <div style='text-align: center; font-size: 500%;font-family: sans-serif;'><b>ASTRA</b></div>
                <p style='text-align: center;'><i>(Advanced Space Technology Roadmap Architecture)</i></p>
                <p style='text-align: center; font-size: 150%;font-family: monospace;'>Studying the technological developments of innovative NASA projects through quantitative methods.</p>
                \n
                """, unsafe_allow_html=True)
    st.text("")
    _, start_btn, about_btn, _ = st.columns([2, 2, 2, 2])
    with start_btn:
        if st.button('Get started', type='primary', use_container_width=True):
            st.session_state.sidebar_state = 'expanded' if st.session_state.sidebar_state == 'collapsed' else 'collapsed'
            st.rerun()
    with about_btn:
        if st.button("Learn more", use_container_width=True):
            st.switch_page('pages/about.py')

    st.markdown("""
                ***
                <p style='text-align: center; font-size: 120%;font-family: monospace;'>Collaboration between</p>
                """, unsafe_allow_html=True)
    _, img, _ = st.columns([1, 1, 1])
    with img:
        st.image('images/logo.png', width=False)
    st.markdown('***')

    with st.container(border=True):
        st.markdown("""
                <p style='text-align: center; font-size: 100%;font-family: monospace;'>DATA ANALYSIS</p>
                """, unsafe_allow_html=True)
        explanation, img, = st.columns([3, 4])
        with explanation:
            st.markdown("""
                        #### Retrieve NASA Technology Project Data

                        Compile a dataset from Techport and SBIR that outlines the features of technology
                        projects including project dates, costs, locations, technology readiness levels (TRLs),
                        taxonomy levels, etc.

                        Additionally, use [pygwalker](https://github.com/Kanaries/pygwalker) as an interactive tool to visualize the dataset.
                        """)
            if st.button("Try it out", use_container_width=True):
                st.switch_page('pages/data.py')
        with img:
            st.image("images/data_viz.png",
                     use_container_width=True, caption='Data visualization of Techport and SBIR data')

    with st.container(border=True):
        st.markdown("""
                <p style='text-align: center; font-size: 100%;font-family: monospace;'>PORTFOLIO THEORY</p>

                #### Automate Portfolio Investment Decisions
                """, unsafe_allow_html=True)
        st.markdown("""
                    Explore approaches designed to automate technology project investment decisions using 
                    two powerful tools: Markowitz Portfolio Theory (MPT) on the left and Deep Reinforcement 
                    Learning (DRL) on the right. These tools have already proven successful in optimizing 
                    financial stock portfolios, and we aim to extend them to technology investments, 
                    enhancing decision-making with a quantitative perspective.
                    """)
        col1, col2 = st.columns(2)
        with col1:
            st.image("images/mpt.png",
                     caption='Markowitz Portfolio Theory', use_container_width=True)
        with col2:
            st.image("images/drl.png",
                     caption='Deep Reinforcement Learning', use_container_width=True)
        if st.button("See investment decisions example based on Techport/SBIR data"):
            st.switch_page('./pages/portfolio.py')
