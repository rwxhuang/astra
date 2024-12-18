import streamlit as st
from pygwalker.api.streamlit import StreamlitRenderer
from st_files_connection import FilesConnection

from src.data_collection import get_astra_data

import pandas as pd

st.set_page_config(
    layout="wide", initial_sidebar_state="collapsed", page_icon='🛰️')

st.header("Technology Project Data")

st.write("Given a search term, web-scrape the Techport database. Put description here...")
search_input = st.text_input('Enter your search input', value="")

if search_input:
    conn = st.connection('s3', type=FilesConnection)
    df = conn.read("astra-data-bucket/techport_all.csv",
                   input_format="csv", ttl=600)

    st.download_button(
        "Press to Download Data",
        df.to_csv(index=False).encode('utf-8'),
        "astra_data.csv",
        "text/csv",
        key='download-csv'
    )

    pyg_app = StreamlitRenderer(df)

    pyg_app.explorer()
