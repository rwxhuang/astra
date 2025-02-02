import streamlit as st
from pygwalker.api.streamlit import StreamlitRenderer

from src.data_collection import get_astra_data

import pandas as pd
import json

# Initialize search_input in session state
if 'search_input' not in st.session_state:
    st.session_state.search_input = ""

# Body of the page
st.header("Technology Project Data")

st.write("Given a search term, web-scrape the Techport database. Put description here...")
st.session_state.search_input = st.text_input(
    'Enter your search input', value=st.session_state.search_input)
search_options = ['heliophysics',
                  'lidar',
                  'earth observation',
                  'artificial intelligence',
                  'robotics',
                  'mars']
cols = st.columns(6)
for i, option in enumerate(search_options):
    with cols[i]:
        if st.button(option, use_container_width=True, icon=':material/search:', type='tertiary'):
            st.session_state.search_input = option
            st.rerun()

# Retrieve data given search input
with st.spinner(f'Fetching data for "{st.session_state.search_input}" from database...'):
    df = get_astra_data(st.session_state.search_input)

st.download_button(
    'Download Data',
    df.to_csv(index=False).encode('utf-8'),
    'astra_data_' + st.session_state.search_input + '.csv',
    'text/csv',
    key='download-csv',
    type='primary',
    icon=':material/download:',
    help='Download the data as a CSV file.',
)

# Read the JSON file for chart specifications
with open('utils/data_charts.json') as file:
    chart_spec = json.load(file)
pyg_app = StreamlitRenderer(df, spec=chart_spec)
pyg_app.explorer()
