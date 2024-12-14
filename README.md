# MIT ASTRA
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <img src="images/NASA_logo.png" alt="Logo" height="80">

  <h3 align="center">ASTRA Markowitz Portfolio Theory (MPT) Dashboard</h3>

  <p align="center">
    Automating portfolio investments for NASA technology roadmapping.
    <br />
    <br />
    <a href>View Demo</a>
    ·
    <a href="https://github.com/rwxhuang/astra-mpt-dashboard/issues">Report Bug</a>
    ·
    <a href="https://github.com/rwxhuang/astra-mpt-dashboard/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#local-dev">Local Development</a></li>
      </ul>
    </li>
    <li>
      <a href="#documentation">Documentation</a>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project
Advanced Space Technology Roadmap Architecture (ASTRA) is a collaborative project between MIT and NASA to meet the following six objectives:

1. <u>Map</u> the NASA technology portfolio to ongoing, planned and potential future missions using a systems approach compatible with NASA’s technology taxonomy (TX01-TX17)
2. <u>Quantify</u> the mission targets achievable as a function of technological figures of merit including sensitivity analysis enabled by integrated modeling and parametric simulation
3. <u>Technology valuation</u> and ranking of the existing technology portfolio (both centrally and in MDs) using TechPort, SMEs and the mission net-present-value (MNPV) concept
4. <u>Portfolio construction</u> by applying a TRL-risk-adapted Markowitz portfolio approach which generates a range of non-dominated space technology investment portfolios
5. <u>Demonstrate</u> the usefulness of the ASTRA methodology by applying it to at least one case study from each MD and prototyping how its outputs could be integrated with and shown in NASA’s environment and tools such as TechPort
6. <u>Disseminate</u> the importance of rigorous technology portfolio investment management using a combination of in-person workshops and online education classes

This project focuses on bulletpoint 4 where we apply Markowitz Portfolio Theory (MPT) to a dataset of publicly availablely NASA projects to yield efficient frontiers of technology portfolio investments and provide insights for decisions. 


<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* Python [streamlit, pandas, PyPortfolioOpt]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Local Development

1. Setup the repo
   ```sh
   git clone https://github.com//rwxhuang/astra-mpt-dashboard.git
   cd astra-mpt-dashboard
   ```
2. Download necessary packages
    ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
      ```
3. Run application locally
   ```sh
   python3 -m streamlit run ASTRA.py
     ```

<!-- DOCUMENTATION -->
## Documentation

To be added

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Roderick Huang - rwxhuang@mit.edu
<p align="right">(<a href="#readme-top">back to top</a>)</p>