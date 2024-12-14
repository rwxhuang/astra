from hydralit import HydraApp
import streamlit as st
from apps import *

st.set_page_config(page_title='MIT ASTRA',page_icon="🛰️",layout='wide',initial_sidebar_state='auto',)

if __name__ == '__main__':

    #this is the host application, we add children to it and that's it!
    app = HydraApp(
        title='Secure Hydralit Data Explorer',
        favicon="🐙",
        hide_streamlit_markers=False,
        use_navbar=True, 
        navbar_sticky=True,
        navbar_animation=True,
    )

    # #Home button will be in the middle of the nav list now
    app.add_app("", icon="🏠", app=HomeApp(title='Home'),is_home=True)

    # #add all your application classes here
    app.add_app("Download Data", icon="📊", app=DownloadDataApp(title="Download Data"))
    app.add_app("Markowitz Portfolio Theory", icon="📈", app=MPTApp(title="Markowitz Portfolio Theory"))
    app.add_app("Deep Reinforcement Learning", icon="🤖", app=DRLApp(title="Deep Reinforcement Learning"))
    app.add_app("About", icon="❔", app=AboutApp(title="About"))
    app.add_app("References", icon="📚", app=ReferencesApp(title="References"))


    app.run()
