import streamlit as st
from pygwalker.api.streamlit import StreamlitRenderer
import pandas as pd

st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_icon='🛰️')

st.header("Technology Project Data")

df = pd.read_csv("./data/techport_all.csv")
 
pyg_app = StreamlitRenderer(df)
 
pyg_app.explorer()