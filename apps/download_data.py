import streamlit as st
from hydralit import HydraHeadApp

class DownloadDataApp(HydraHeadApp):

    def __init__(self, title = '', **kwargs):
        self.__dict__.update(kwargs)
        self.title = title

    def run(self):
        self._body()

    def _body(self):
        st.write("# Downloading Data")