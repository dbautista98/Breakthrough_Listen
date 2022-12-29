# Breakthrough_Listen

email: dbautista98@berkeley.edu

Created this github repository to save and keep track of my work with Breakthrough Listen

## Setup

To run code from this repo: 

First clone this repo to your machine 
```
git clone https://github.com/dbautista98/Breakthrough_Listen.git
```
Set up a `conda` environment using the `.yml` file

```
conda env create -f turboSETI.yml
```

Activate the new conda environment after everything installs

```
conda activate turboSETI
```

Note: if you want to name this conda environment something other than `turboSETI`, change the first line of `turboSETI.yml` from `name: turboSETI` to `name: new_name_here`

## `spectral_occupacy.py` Usage

`spectral_occupancy` is a program to visualize how often `turboSETI` detects narroband signals in a given frequency bin. Spectral Occupancy is defined as the fraction of observations in which at least `turboSETI` detects *at least* one narrowband signal. To calculate this, a large number of `.dat` files are needed. 

With large datasets, it can be helpful to group the data with similar parameters, so this program has options to split observations based on the time of day the data was collected, the altitude the telescope was angled, and the azimuth it was angled. It also has the option to split the data up into smaller patches of the sky based on the altitude and azimuth.

To see the input options for this program, call

```
python3 spectral_occupancy.py --help
```

Positional (required) arguments:
* `band` -  the observing band from Green Bank Telescope. Choose from `{L,S,C,X}` 
* `folder` - the path to the directory containing the `.dat` files. 

Optional arguments:
* `-notch_filter` - flag to exclude the data from {L,S} band notch filters.
* `-t` to pass in a `.txt` file with the paths, if your `.dat` files are contained in multiple directories. 
* `-outdir` or `-o` to specify the path that you want the figures.
* `-width` or `-w` - the width of a frequency bin in MHz. The default value is 1 MHz wide bins.
* `-save` or `-s` - saves the histogram and bin edge data so it can be used in other programs.
* `-lower_time` or `-lt` - sets the lower time boundary for splitting the data based on time of day. The time zone is GMT-5, the timezone of Green Bank Telescope.
* `-upper_time` or `-ut` - sets the upper time boundary for splitting the data based on time of day. To be used in combination with `-lower_time`
* `-lower_alt` or `-la` - sets the lower altitude boundary for splitting the data.
* `-upper_alt` or `-ua` - sets the upper altitude boundary for splitting the data. To be used in combination with `-lower_alt`
* `-lower_az` or `-lz` - sets the lower azimuth boundary for splitting the data.
* `-upper_az` or `-uz` - sets the upper azimuth boundary for splitting the data. To be used in combination with `-lower_az`
* `-N_altitude_bins` or `-na` - number of bins to split the altitude range of [0,90] into.
* `-N_azimuth_bins` or `-nz` - number of bins to split the azimuth range [0,360) into. To be used with `-N_altitude_bins`
* `-no_default` or `-no` - flag to tell the program $not$ to plot the default spectral occupancy with zero grouping of the data. This is best used with splitting the data, if you have already plotted the regular spectral occupancy.
* `-exclude_zero_drift` or `-exclude` - flag to tell the program to exclude the hits with a drift rate of zero. This is best used if you have a mix of old and new `turboSETI` outputs, as older versions did not automatically discard zero drift hits. Recommend to use this option if you are unsure. 

Below is an example of calling this program on a directory of L band`.dat` files. This inlcudes the optional parameters telling the program to exclude the data from the notch filter region and split the files up into north and south halves of the sky, then save the figures in another directory:

```
python3 spectral_occupancy.py L /home/path/to/dats/ -o /home/output/path/here/ -lz 90 -uz 270 -nf
```

## `remove_DC_spike.py` Usage

`remove_DC_spike` is a program that goes through a `.dat` file an removes the coarse channel center frequencies. These are problematic because they are not physically meaningful and show up in spectral occupancy plots as a bin with 100% occupancy that repeats every ~3MHz from the starting frequency. They arise as a result of the zero frequency power during a fourier transform. 

This program was initially written to be called prior to running `spectral_occupancy.py`, but is now called as `spectral_occupancy` runs. It isn't necessary to call this as a stand alone program anymore, but if you want, the steps are below:

To see the input options for this program, call

```
python3 remove_DC_spike.py --help
```

Positional (required) arguments:

* `band` -  the observing band from Green Bank Telescope. Choose from `{L,S,C,X}`  

Optional arguments:

* `folder` - the path to the directory containing the `.dat` files.
* `-t` to pass in a `.txt` file with the paths, if your `.dat` files are contained in multiple directories. 
* `-outdir` or `-o` to specify the path that you want the cleaned `.dat` files saved to

An example of calling this program on a directory of `.dat` files and saving the cleaned files in another directory is:

```
python3 remove_DC_spike.py L /home/path/to/dats/ -o /home/output/path/here/
```

## `candidate_ranking.py` Usage

To run this program, you will first need to be in the directory it is in. 

```
cd Breakthrough_Listen/candidate_ranking/
```

This program uses the spectral occupancy data from a large collection of observations to rank `turboSETI` candidates as how rare a hit in that frequency bin is. A ranking of one means that no observation has detected a hit at that frequency bin before, while a ranking of zero means that every observation records hits at that frequency bin. This ranking can be used as a way to tell us how interested we should be in a candidate event. 

To see the input options for this program, call

```
python3 candidate_ranking.py --help
```

Positional (required) arguments:

* `band` -  the observing band from Green Bank Telescope. Choose from `{L,S,C,X}` 

Optional arguments:
* `-folder` or `-f` - the path to the directory containing the event `.csv` files. 
* `-text` or `-t` - path to a text file containing the filepaths to the event `.csv` files
* `-outdir` or `-o` - directory to save the outputs to
* `-grid_ranking` or `-g` - flag to rank candidates based on observations from the same region of sky

Below is an example of how to call the program and save outputs to a different directory:

```
python3 candidate_ranking.py L -f /home/path/to/files/ -o /home/output/path/here/
```

## `flag_bad_files.py` Usage

This program passes through spliced `.dat` files and identifies fiels that are affected by compute node dropout, identifiable by a total absence of data in an interval of 187.5 MHz from the start of the spectrum. This program also checks the data for an unusually high number of hits, which is indicative of a satellite passing near the telescope beam and introducing spectral leakage. 

To see the input options for this program, call

```
python3 flag_bad_files.py --help
```

Positional (required) arguments:
* `band` -  the observing band from Green Bank Telescope. Choose from `{L,S,C,X}` 
* `data_dir` - filepath to the directory where the `.dat` files are stored
* `algorithm` - choose from `{turboSETI,energy_detection}` as the input. This code will work on output files from either algorithm, but `energy_detection` output `.csvs` are quite large, so you will probably only be using `turboSETI` files
* `-outdir` or `-o` - directory to save the outputs to
* `-threshold` or `-t` - This argument is only used on `energy_detection` files. This sets the threshold below which all hits will be excluded. The default threshold is set to 4096

Below is an example of how to call this progam on a directory of spliced `.dat` files and save the outputs to a different directory:

```
python3 flag_bad_files.py L /home/path/to/files/ turboSETI -o /home/output/path/here/
```

## Other useful things:

`plot_dat_demo.ipynb` - a demonstration of how to use `turbo_seti.find_event.plot_dat`, a program that I wrote to plot a visualization of the hits and events that `turbo_seti` is detecting 

`bldw_tools` - some python scripts that I wrote to use `bldw` to dig for `.dat` files

### Fall 2020

This fall I wrote a [program](https://github.com/dbautista98/Breakthrough_Listen/blob/main/spectral_occupancy.py) to calculate the spectral occupancy of data collected at Green Bank Telescope.  During this process, I needed to remove signals corresponding to DC spike channels, which I also wrote a [program](https://github.com/dbautista98/Breakthrough_Listen/blob/main/remove_DC_spike.py) for. 

Here is a jupyter notebook demonstrating the results of these programs: https://github.com/dbautista98/Breakthrough_Listen/blob/main/Final_Spectral_Occupancy.ipynb


### Spring 2021 

This semester I started with writing a program to help visualize what turboSETI is “seeing” as it processes a cadence. I wrote a function called [plot_dat](https://github.com/UCBerkeleySETI/turbo_seti/blob/master/turbo_seti/find_event/plot_dat.py), which generates a plot similar to those generated via plot_event_pipeline, but instead of showing only an overlay of the detected candidate, it also shows the hits detected during the doppler drift search, regardless of whether the hit was part of a candidate signal, or just a false positive “hit.” Here is a link to a jupyter notebook demonstrating how to use plot_dat, starting with an on/off cadence, running a turboSETI doppler drift search on it, and then plotting the detected hits via plot_dat. https://github.com/dbautista98/Breakthrough_Listen/blob/main/plot_dat_demo.ipynb 

In the latter part of the semester, I resumed working on long-term GBT statistics, with the goal to come up with a method to quantify “good” and “bad” GBT data. This involves gathering a large number of .dat files that can be used to get a long term average that individual .dat files can be compared to. Comparing the file to the long-term data allows us to determine if there are parts of the spectrum that have an unusually high or low number of hits detected, which could be indicative of a high amount of RFI, or a problem with the collecting/storing of the data. My current progress on the long-term GBT statistics can be found here: https://github.com/dbautista98/Breakthrough_Listen/blob/main/long_term_statistics.ipynb

PS: if you have some .dat files that you are willing to share, feel free to reach out to me on Slack or at my email above :)


### Fall - Spring 2021

This semester I am working on estimating the likelihood that a signal flagged as interesting by turboSETI is a false positive and is instead due to a satellite passing over the telescope. The data I am using for this analysis was collected at Green Bank Telescope and initially used by Raffy Traas. To do this comparison, I will start by comparing the results of turboSETI to the results of Yuhong's [energy detection](https://github.com/FX196/SETI-Energy-Detection) algorithm. To do this, I am generating spectral occupancy plots using data from turboSETI's dat file outputs and Energy Detection's csv outputs. 
