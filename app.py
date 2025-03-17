import streamlit as st
from st_pages import get_nav_from_toml

if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'collapsed'

nav = get_nav_from_toml()

pg = st.navigation(nav)
pg.run()
