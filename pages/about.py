import streamlit as st

st.set_page_config(layout="centered",
                   initial_sidebar_state="collapsed", page_icon='🛰️')

st.markdown(
    """
    # About ASTRA
    Advanced Space Technology Roadmap Architecture (ASTRA) is a collaborative project between MIT and NASA to meet the following six objectives:
    1. **Map** the NASA technology portfolio to ongoing, planned and potential future missions using a systems approach compatible with NASA’s technology taxonomy (TX01-TX17)
    2. **Quantify** the mission targets achievable as a function of technological figures of merit including sensitivity analysis enabled by integrated modeling and parametric simulation
    3. **Technology** valuation and ranking of the existing technology portfolio (both centrally and in MDs) using TechPort, SMEs and the mission net-present-value (MNPV) concept
    4. **Portfolio** Construction by applying a TRL-risk-adapted Markowitz portfolio approach which generates a range of non-dominated space technology investment portfolios
    5. **Demonstrate** the usefulness of the ASTRA methodology by applying it to at least one case study from each MD and prototyping how its outputs could be integrated with and shown in NASA’s environment and tools such as TechPort
    6. **Disseminate** the importance of rigorous technology portfolio investment management using a combination of in-person workshops and online education classes
    # People
"""
)

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(height=500, border=True):
        st.image("./images/oli.jpg", use_container_width=True)
        st.markdown("""#### Olivier L. de Weck \n *Professor*: Apollo Program Professor of Astronautics, Professor of Engineering Systems, ESL Faculty Director""")
    with st.container(height=500, border=True):
        st.image("./images/julia.jpg", use_container_width=True)
        st.write(
            "#### Julia Milton \n *PhD 2023, Aeronautics and Astronautics*: SEAri")
    with st.container(height=500, border=True):
        st.image("./images/nadiak.png", use_container_width=True)
        st.write(
            "#### Nadia Khan \n *Graduate Researcher*: Technology Policy Program Masters Candidate")
    with st.container(height=500, border=True):
        st.image("./images/aaron_martin.png", use_container_width=True)
        st.write(
            "#### Aaron Martin \n *Undergraduate Student*: Double majoring in Mathematics and Computer Science")

with col2:
    with st.container(height=500, border=True):
        st.image("./images/afreen.jpeg", use_container_width=True)
        st.write(
            "#### Afreen Siddiqi \n *Research Scientist*: Strategic Engineering Research Group")
    with st.container(height=500, border=True):
        st.image("./images/alex.jpg", use_container_width=True)
        st.write(
            "#### Alex Koenig \n *SM 2023, Aerospace*: Strategic Engineering Research Group")
    with st.container(height=500, border=True):
        st.image("./images/hailey.jpg", use_container_width=True)
        st.write(
            "#### Hailey D Polson \n *Undergraduate Student*: Majoring in Aerospace Engineering (16)")
    with st.container(height=500, border=True):
        st.image("./images/hailey.jpg", use_container_width=True)
        st.write(
            "#### Rachel Yang \n *Undergraduate Student*: Majoring in AI and Decision Making (6-4)")

with col3:
    with st.container(height=500, border=True):
        st.image("./images/george.jpeg", use_container_width=True)
        st.markdown(
            """#### George Lordos \n *Research Scientist*: Strategic Engineering Research Group""")
    with st.container(height=500, border=True):
        st.image("./images/chloe.jpg", use_container_width=True)
        st.write("#### Chloe Gentgen \n *Graduate Researcher*: Strategic Engineering Research Group, Ph.D. Candidate, MIT AeroAstro")
    with st.container(height=500, border=True):
        st.image("./images/roderick.jpg", use_container_width=True)
        st.write(
            "#### Roderick Huang \n *MEng Student*: Double majored in Mathematics (18) and Computer Science (6-3)")