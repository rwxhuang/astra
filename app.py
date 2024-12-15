import streamlit as st
from utils.utils import footer
from st_pages import add_page_title, get_nav_from_toml

if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'collapsed'

nav = get_nav_from_toml()

st.logo("images/logo.png")

pg = st.navigation(nav)
pg.run()

footer()
