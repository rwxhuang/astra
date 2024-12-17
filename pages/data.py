import streamlit as st
from pygwalker.api.streamlit import StreamlitRenderer
from st_files_connection import FilesConnection
import pandas as pd

st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_icon='🛰️')

st.header("Technology Project Data")

conn = st.connection('s3', type=FilesConnection)
df = conn.read("astra-data-bucket/techport_all.csv", input_format="csv", ttl=600)
 
pyg_app = StreamlitRenderer(df)
 
pyg_app.explorer()