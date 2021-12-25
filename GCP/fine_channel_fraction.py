import numpy as np
import pandas as pd
import spectral_occupancy as so
import glob
from tqdm import trange
import argparse
import pickle

# want to loop over the data and do the following
# break the data up into 1MHz bins? and find the 
# fraction of fine channels that have a hit in them

def read_data(csv_path, header_path, threshold=4096):
    # read in data
    tbl = pd.read_csv(csv_path)

    # remove DC spikes
    tbl = so.remove_DC_spikes(tbl, header_path)

    # filter above threshold
    tbl = tbl.iloc[np.where(tbl["statistic"] > threshold)]
    return tbl

def split_data(tbl, start, end):
    mask = np.where((tbl["freqs"] >= start) & (tbl["freqs"] < end))
    return tbl.iloc[mask]

def interval_fraction(tbl, start, stop, fine_channel_width=1e-6):
    tbl = split_data(tbl, start, stop)
    bins = np.linspace(start, stop, int((stop-start)/fine_channel_width)+1, endpoint=True) # the +1 is to make the number of bins correct after constructing histogram
    hist, bin_edges = np.histogram(tbl["freqs"], bins=bins)
    fraction = np.sum((hist > 0))/len(hist)
    return fraction

def one_file(csv_data, GBT_band, bin_width=1, fine_channel_width=1e-6):
    # band boundaries as listed in Traas 2021
    if GBT_band=="L":
        min_freq = 1100
        max_freq = 1900
    if GBT_band=="S":
        min_freq = 1800
        max_freq = 2800
    if GBT_band=="C":  
        min_freq = 4000
        max_freq = 7800
    if GBT_band=="X":
        min_freq = 7800
        max_freq = 11200
    spectral_occupancy_bins = np.arange(min_freq, max_freq+0.5*bin_width, bin_width)
    bin_fractions = np.empty_like(spectral_occupancy_bins)
    for i in range(len(spectral_occupancy_bins)):
        bin_fractions[i] = interval_fraction(csv_data, spectral_occupancy_bins[i], spectral_occupancy_bins[i] + 1, fine_channel_width=fine_channel_width)
    return bin_fractions


if __name__ == "__main__":
    fine_channel_width = 3/(2**20) # Lebofsky et al 2019 for HSR (high spectral resolution) files 
    parser = argparse.ArgumentParser(description="determines the fraction of fine channels that have hits in a spectral occupancy bin")
    parser.add_argument("band", help="the GBT band that the data was collected from. Either L, S, C, or X")
    parser.add_argument("folder", help="directory containing the folders which store the energy detection output")
    parser.add_argument("-width", "-w", help="width of the spectral occupancy bin in Mhz", type=float, default=1)
    parser.add_argument("-threshold", "-t", help="threshold below which all hits will be excluded. Default is 4096", type=float, default=4096)
    parser.add_argument("-fine", "-f", help="width of fine channel in MHz. This needs to be updated to have the correct fine channel width. Default width is 3MHz/2^20 for HSR files per Lebofsky et al 2019", type=float, default=fine_channel_width)
    args = parser.parse_args()

    files = glob.glob(args.folder+"/*")
    dataset_fractions = 0
    n_files = len(files)
    print("Processing files...")
    for i in trange(len(files)):
        data = (files[i]+"/all_info_df.csv")
        header = (files[i] + "/header.pkl")
    
        tbl = read_data(data, header, threshold=args.threshold)
        bin_fractions = one_file(tbl, args.band)
        dataset_fractions += bin_fractions

    # normalize data by number of files
    dataset_fractions = dataset_fractions/n_files

    print("Saving fine channel data")
    to_save = {"fine channel fraction":dataset_fractions, "bin_width":args.width, "fine_channel_width":args.fine, "band":args.band, "threshold":args.threshold, "algorithm":"energy detection", "n files":len(files)}
    filename = "energy_detection_%s_band_fine_channel_fraction_.pkl"%(args.band)
    with open(filename, "wb") as f:
        pickle.dump(to_save, f)

    print("Done!")