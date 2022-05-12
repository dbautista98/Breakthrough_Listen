# TESS_statistics

The data stored in this directory was calculated from the `.h5` files found in the `bl_tess` Google Cloud storage bucket. This data was collected from Green Bank Telescope between the dates of September 18, 2019 and March 26, 2020. These `.h5` files were passed through the [turboSETI](https://github.com/UCBerkeleySETI/turbo_seti) and [energy detection](https://github.com/UCBerkeleySETI/BL-Reservoir/tree/master/energy_detection) (further explanation [here](https://github.com/FX196/SETI-Energy-Detection/blob/master/README.md)) algorithms. 

Due to the nature and size of the energy detection outputs, I applied a minimum `statistic` threhsold of 4096, which gives a similar spectral occupancy to that of turboSETI. Then I calculated a histogram of the hits in each observation, with bin widths of 1 MHz, and stacked them all in a single `.csv` file, with each line corresponding to a different observation. 
