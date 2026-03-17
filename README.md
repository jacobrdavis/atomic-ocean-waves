# atomic-ocean-waves
Analysis of ocean surface wave data collected during the Atlantic Tradewind Ocean–Atmosphere Mesoscale Interaction Campaign (ATOMIC).

# Getting started

1. Clone this repository.  In the terminal, run:
   ```sh
   git clone https://github.com/jacobrdavis/atomic-ocean-waves.git
   ```
2. Download the data and move it to [data_input/](data_input/) (see the README inside the folder).
3. Create a Python environment.  If using conda/mamba, run:
   ```sh
   conda env create -f environment.yml
   ```
   This will install the necessary dependencies and an editible version of the package source code inside [src/](src/).
4. Activate the new environment and run the .ipynb notebooks inside [notebooks/](notebooks/) in order. These notebooks will process the raw data inside [data_input/](data_input/) and reproduce the analysis and figures.
