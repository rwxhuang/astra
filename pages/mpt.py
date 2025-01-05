import streamlit as st

st.set_page_config(
    layout="wide", initial_sidebar_state="expanded", page_icon='🛰️')

st.header("Markowitz Portfolio Theory")

with st.sidebar:
    st.header('Data Upload')
    st.file_uploader(
        'Upload technology project data'
    )
    st.header('Projects to Invest In')
    st.file_uploader(
        'Upload technology projects to invest in'
    )
    with st.expander('Optional custom modifications', expanded=False):
        st.slider('slider', min_value=0., max_value=1., step=0.01)
