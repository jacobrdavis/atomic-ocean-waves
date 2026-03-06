# data_input/

This folder should contain ATOMIC wave datasets including the SWIFT buoys, Wave Glider and Saildrone platforms, shipboard RIEGL LiDAR, airbourne Wide Swath Radar Altimeter, and WAVEWATCH III simulations.

These datasets are publicly archived on the ATOMIC / EUREC4A Data Information and Access page https://www.psl.noaa.gov/atomic/data/. Repositories specific to this work are provided below:


| platform     | description                                          | DOI                                 |
|--------------|------------------------------------------------------|-------------------------------------|
| P3           | Aircraft Remote Sensing Cloud, Rain, Wind, Wave      | https://doi.org/10.25921/x9q5-9745  |
| RIEGL        | LiDAR-measured waves                                 | https://doi.org/10.25921/etxb-ht19  |
| Saildrone    | Wave and meteorology                                 | https://doi.org/10.25921/9km0-f614  |
| Ship         | Fluxes, Surface Ocean and Meteorology, Navigation    | https://doi.org/10.25921/etxb-ht19  |
| SWIFT        |  Wave and meteorology                                | https://doi.org/10.25921/s5d7-tc07  |
| Wave Glider  | Wave and meteorology                                 | https://doi.org/10.25921/dvys-1f29  |
| WSRA         | Aircraft Remote Sensing Wave, Rain                   | https://doi.org/10.25921/qm06-qx04  |
| WW3          | TODO:                                                | TODO:                               |

Download the datasets and unzip the contents into the `data` folder. It should have the following structure:

```
data_input/
├── P3/
├── riegl/
├── saildrone/
├── ship/
├── SWIFT/
├── wave_glider/
├── WSRA/
└── WW3/
```
