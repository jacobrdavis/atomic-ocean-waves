# atomic-ocean-waves
Analysis of ocean surface wave data collected during the Atlantic Tradewind Ocean–Atmosphere Mesoscale Interaction Campaign (ATOMIC).

# Getting started

1. Clone this repository.  In the terminal, run:
   ```sh
   git clone https://github.com/jacobrdavis/neural-network-based-methods-for-das-ocean-surface-wave-measurement.git
   ```
3. Download the data and move it to [data_input/](data_input/) (see the README inside the folder).
4. Create a Python environment.  If using conda/mamba, run:
   ```sh
   conda env create -f environment.yml
   ```
5. Install the source code. Access the latest release at [https://github.com/jacobrdavis/atomic-ocean-waves/releases/latest](https://github.com/jacobrdavis/atomic-ocean-waves/releases/latest) and install it, e.g., using pip:
   ```sh
   pip install https://github.com/jacobrdavis/atomic-ocean-waves/releases/download/v2026.03.0/atomic_ocean_waves-2026.3.0.tar.gz
   ```
6. Run the .ipynb notebooks inside [notebooks/](notebooks/) in order. These notebooks will process the raw data inside [data_input/](data_input/) and reproduce the analysis and figures.
