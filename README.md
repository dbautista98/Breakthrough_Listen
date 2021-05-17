# Breakthrough_Listen

Created this github repository to save and keep track of my work with Breakthrough Listen

## Fall 2020

This fall I wrote a [program](https://github.com/dbautista98/Breakthrough_Listen/blob/main/spectral_occupancy.py) to calculate the spectral occupancy of data collected at Green Bank Telescope.  During this process, I needed to remove signals corresponding to DC spike channels, which I also wrote a [program](https://github.com/dbautista98/Breakthrough_Listen/blob/main/remove_DC_spike.py) for. 

Here is a jupyter notebook demonstrating the results of these programs: https://github.com/dbautista98/Breakthrough_Listen/blob/main/Final_Spectral_Occupancy.ipynb


## Spring 2021 

This semester I started with writing a program to help visualize what turboSETI is “seeing” as it processes a cadence. I wrote a function called [plot_dat](https://github.com/UCBerkeleySETI/turbo_seti/blob/master/turbo_seti/find_event/plot_dat.py), which generates a plot similar to those generated via plot_event_pipeline, but instead of showing only an overlay of the detected candidate, it also shows the hits detected during the doppler drift search, regardless of whether the hit was part of a candidate signal, or just a false positive “hit.” Here is a link to a jupyter notebook demonstrating how to use plot_dat, starting with an on/off cadence, running a turboSETI doppler drift search on it, and then plotting the detected hits via plot_dat. https://github.com/dbautista98/Breakthrough_Listen/blob/main/plot_dat_demo.ipynb 

In the latter part of the semester, I resumed working on long-term GBT statistics, with the goal to come up with a method to quantify “good” and “bad” GBT data. This involves gathering a large number of .dat files that can be used to get a long term average that individual .dat files can be compared to. Comparing the file to the long-term data allows us to determine if there are parts of the spectrum that have an unusually high or low number of hits detected, which could be indicative of a high amount of RFI, or a problem with the collecting/storing of the data. My current progress on the long-term GBT statistics can be found here: https://github.com/dbautista98/Breakthrough_Listen/blob/main/long_term_statistics.ipynb

PS: if you have some .dat files that you are willing to share, feel free to reach out to me on Slack or at my email above :)
