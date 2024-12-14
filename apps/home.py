import streamlit as st
from hydralit import HydraHeadApp

class HomeApp(HydraHeadApp):
    def __init__(self, title = '', **kwargs):
        self.__dict__.update(kwargs)
        self.title = title

    def run(self):
        self._body()

    def _body(self):
        _, body, _ = st.columns([1,4,1])
        with body: 
            st.markdown("""
                        <p style='text-align: center; font-size: 500%;font-family: sans-serif;'><b>MIT ASTRA</b></p>
                        <p style='text-align: center; font-size: 150%;font-family: sans-serif;'>Studying the technological developments of innovative NASA projects through quantitative methods.</p>
                        """, unsafe_allow_html=True)
            _, button, _ = st.columns([3,4,3])
            with button:
                st.button("Learn more", type='primary', use_container_width=True)
        

        