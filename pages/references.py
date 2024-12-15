import streamlit as st

st.set_page_config(layout="centered", initial_sidebar_state="collapsed", page_icon='🛰️')

st.markdown(
    """
    # References
    #### Listed below are papers and books written by the ASTRA team.
    ##### Markowitz Portfolio and Technology Valuation Methodologies
    - [Book] [Technology Roadmapping and Development - A Quantitative Approach to the Management of Technology](https://link.springer.com/book/10.1007/978-3-030-88346-1); Olivier de Weck
    - [Paper] Siddiqi A., Milton J., Khan N. and de Weck O.L. (2023), ["Application of Markowitz Portfolio Theory for Space Technologies”](https://iafastro.directory/iac/paper/id/79995/summary/), 74th International Astronautical Congress (IAC), Baku, Azerbaijan
    - [Book Chapter] Lordos, G.C., Hoffman, J.A., de Weck, O.L. (2023). [Lifetime Embodied Energy: A Theory of Value for the New Space Economy](https://doi.org/10.1007/978-3-030-97913-3_32). In: Badescu, V., Zacny, K., Bar-Cohen, Y. (eds) Handbook of Space Resources. Springer, Cham. 
    ##### Heliophysics Case Study
    - [Paper] Shtofenmakher A., Miller D., Waters B. et al., (2024) "SUNFLOWER Observatory: Heliocentric Satellite Constellation for Continuous Space Weather Monitoring”, 2024 IEEE Aerospace Conference, Big Sky, MT USA, 2024
    ##### Astrophysics Case Study
    - [SM Thesis] Koenig, A. (2023) [“A Systems Framework for Multi-Messenger Astronomy”](https://hdl.handle.net/1721.1/154207), SM Thesis, Massachusetts Institute of Technology, Cambridge, MA
    ##### Earth Observation Case Study
    - [Paper] [Trends And Technology Roadmapping In Earth Observation Missions](https://drive.google.com/file/d/1vITfJxRkSWXGFOPaLDRtF7-y1qtPkGGb/view);
        Afreen Siddiqi, Julia Milton, George Lordos, Olivier de Weck   *IGARSS 2022*
    - [Paper] [Performance Trajectories Of Earth Observation Technologies: Parametrized Formulations And Estimation](https://drive.google.com/file/d/17hjXVWK94YFVfdERHGjm71L3dEgNE7zk/view);
        Afreen Siddiqi, Julia Milton, Olivier de Weck   *IGARSS 2023*
    - [Paper] [Earth Observation Technologies for Climate Change Adaptation and Monitoring: Future Projection from Decadal Trends](https://drive.google.com/file/d/146rDReSWqNHAe6hUhiKoz9Rb2w4GAIP1/view);
        Afreen Siddiqi, Julia Milton, Maria Cabrera, Olivier de Weck    *GLOC 2023*
    ##### Artemis Case Study: Technology System Modeling
    - [Paper] Lordos, G. and de Weck, O.L. ["Technology Roadmapping of an ISRU System for NASA Using a Medium-Fidelity System Dynamics Simulation,"](https://arc.aiaa.org/doi/abs/10.2514/6.2022-4307) AIAA 2022-4307. ASCEND 2022. October 2022.
    ##### Artemis Case Study: Novel Figures of Merit and System Architectures for planetary surface Human exploration
    - [Paper] Lordos G., Hoying M., AlSadah Y. et al., (2023) ["Pale Red Dot: Polis-based Architecture for the Long-term Exploration of the Red Planet”](https://rascal.nianet.org/wp-content/uploads/2023-RASC-AL-Technical-Paper-MIT-Updated.pdf), 2023 NASA RASC-AL Forum, Cocoa Beach, Florida
    - [Abstract] Lordos G., Siddiqi A., Hoying M. et al., (2023) ["A General Model of ISRU Technology Valuation and Technology Portfolio Construction for Crewed Mars Missions”](https://isruinfo.com/public/index.php?page=srr_23), Space Resources Roundtable (SRR) 2023, Golden, Colorado
    - [Paper] Lordos G., Hoying M., McKinney L., de Weck O.L., and Hoffman J.A., (2024) ["Leveraging Economies of Scale and Gains from Specialization for Robust Crewed Mars Architectures”](https://www.researchgate.net/publication/379342061_Leveraging_Economies_of_Scale_and_Gains_from_Specialization_for_Robust_Crewed_Mars_Architectures?_sg%5B0%5D=afZML3FcQu4rmNfsE9jd63Ca7fbldfqPCdtGF5unWUijdglk_op-tbO8WUjDnT8nYRcCJdCu024AEqaEoVRXt98dzQln6X9py6JXRkUY.rGCTsWxMD739LwnWDbjPh7Pra4pUXzUmaIGHsl5tBA97MHiHCK653sePocIXsRU4tyvtNj-gXtFjMAAappYfCw&_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6ImhvbWUiLCJwYWdlIjoicHJvZmlsZSIsInByZXZpb3VzUGFnZSI6InByb2ZpbGUiLCJwb3NpdGlvbiI6InBhZ2VDb250ZW50In19), 2024 IEEE Aerospace Conference, Big Sky, MT USA, 2024
    - [PhD thesis] Lordos G., (2024), ["Engineering Robust and Autocatalytic Architectures for Human Missions to Mars”](https://hdl.handle.net/1721.1/153785), PhD Thesis, Massachusetts Institute of Technology, Cambridge, MA
    ##### Novel Technology Strategies for Robotic Planetary Exploration
    - [Paper] Lordos G., Brown M.J., Latyshev K. et al., ["WORMS: Field-Reconfigurable Robots for Extreme Lunar Terrain,"](https://www.doi.org/10.1109/AERO55745.2023.10115833) 2023 IEEE Aerospace Conference, Big Sky, MT, USA, 2023, pp. 1-21, doi: 10.1109/AERO55745.2023.10115833
    #### Listed below are resources used to build this application.
    - Martin, R. A., (2021). PyPortfolioOpt: portfolio optimization in Python. Journal of Open Source Software, 6(61), 3066, https://doi.org/10.21105/joss.03066
    - https://selenium-python.readthedocs.io/
    - https://github.com/pskrunner14/info-retrieval
""")