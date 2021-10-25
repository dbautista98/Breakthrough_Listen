import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from tqdm import trange
import argparse

def calculate_hist(csv_file, GBT_band, bin_width=1, threshold=2048):
    # read in the data
    tbl = pd.read_csv(csv_file)
    tbl = tbl.iloc[np.where(tbl["statistic"] > threshold)]

    # make the bins of the histogram
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
    bins = np.linspace(min_freq, max_freq, int((max_freq - min_freq)/bin_width), endpoint=True)
    hist, bin_edges = np.histogram(tbl["freqs"], bins=bins)
    del tbl
    return hist, bin_edges

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generates a histogram from csv files outputted from the energy detection algorithm")
    parser.add_argument("band", help="the GBT band that the data was collected from. Either L, S, C, or X")
    parser.add_argument("folder", help="folder where the csv files are stored")
    parser.add_argument("-width", "-w", help="width of bin in MHz, default is 1 MHz", default=1)
    parser.add_argument("-threshold", "-t", help="minimum statistic value that will be preserved, default is 2048", default=2048)
    parser.add_argument("-outdir", "-o", help="directory to store histograms in", default=None)
    args = parser.parse_args()

    print("Gathering folders...", end="")
    folder_paths = glob.glob(args.folder+"/*")
    print("Done gathering %s folders"%len(folder_paths))

    csv_name = "/all_info_df.csv"
    print("Calculating first histogram...", end="")
    total_hist, bin_edges = calculate_hist(folder_paths[0]+csv_name, args.band, bin_width=args.width, threshold=float(args.threshold))
    print("Done.")
    print("Calculating remaining histograms...", end="")
    for i in trange(len(folder_paths)-1):
        hist, edges = calculate_hist(folder_paths[i+1]+csv_name, args.band, bin_width=args.width, threshold=float(args.threshold))
        total_hist += hist
    print("Done.")

    plt.figure(figsize=(20, 10))
    plt.bar(bin_edges[:-1], total_hist)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Count")
    plt.title("%s Band Energy Detection Histogram with n=%s files and threshold=%s"%(args.band, len(folder_paths), args.threshold))
    if args.outdir is not None:
        outdir = args.outdir+"/"
        plt.savefig("%s%s_band_energy_detection_hist_threshold_%s.pdf"%(outdir, args.band, args.threshold))
    else:
        plt.savefig("%s_band_energy_detection_hist_threshold_%s.pdf"%(args.band, args.threshold))
    print("ALL DONE.")
