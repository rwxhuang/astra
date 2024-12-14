import streamlit as st
from hydralit import HydraHeadApp

class MPTApp(HydraHeadApp):

    def __init__(self, title = '', **kwargs):
        self.__dict__.update(kwargs)
        self.title = title

    def run(self):
        self._body()

    def _body(self):
        st.write("# Markowitz Portfolio Theory")
        with st.sidebar:
            st.write('fdsfs')