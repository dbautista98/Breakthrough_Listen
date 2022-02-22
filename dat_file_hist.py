from GCP.compute_hist import calculate_hist
import spectral_occupancy as so
import argparse
import glob
from tqdm import trange
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generates a histogram of the hit counts from a given set of .dat files")
    parser.add_argument("band", help="the GBT band that the data was collected from. Either L, S, C, or X")
    parser.add_argument("folder", help="the directory the .dat files are stored in")
    parser.add_argument("-width", "-w", help="width of bin in MHz", type=float, default=1)
    parser.add_argument("-notch_filter", "-nf", help="exclude data that was collected within GBT's notch filter", action="store_true")
    args = parser.parse_args()

    print("Gathering files...", end="")
    dat_files = glob.glob(args.folder+"/*.dat")
    print("Done.")

    print("removing DC spikes...", end="")
    dat_files = so.remove_spikes(dat_files, args.band)
    print("Done.")

    total_hist, bin_edges = so.calculate_hist(dat_files[0], args.band, bin_width=args.width)

    print("Calculating remaining histograms...", end="")
    for i in trange(len(dat_files)-1):
        hist, edges = so.calculate_hist(dat_files[i+1], args.band, bin_width=args.width)
        total_hist += hist
    print("Done.")

    data_dict = {"frequency":edges[:-1], "count":total_hist}
    df = pd.DataFrame(data_dict)

    if args.band=="L":
        if args.notch_filter:
            print("Excluding hits in the range 1200-1341 MHz")
            df = df[(df["frequency"] < 1200) | (df["frequency"] > 1341)]
    
    if args.band=="S":
        if args.notch_filter:
            print("Excluding hits in the range 2300-2360 MHz")
            df = df[(df["frequency"] < 2300) | (df["frequency"] > 2360)]

    df.to_csv("%s_band_turboSETI_hist.csv"%args.band)

    plt.figure(figsize=(20, 10))
    plt.bar(df["frequency"], df["count"], width=1)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Count")
    plt.title("%s Band turboSETI Histogram with n=%s files"%(args.band, len(dat_files)))
    plt.savefig("%s_band_turboSETI_hist.pdf"%args.band)