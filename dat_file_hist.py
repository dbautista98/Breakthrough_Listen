from GCP.compute_hist import calculate_hist
import spectral_occupancy as so
import argparse
import glob
from tqdm import trange
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generates a histogram of the hit counts from a given set of .dat files")
    parser.add_argument("band", help="the GBT band that the data was collected from. Either L, S, C, or X")
    parser.add_argument("folder", help="the directory the .dat files are stored in")
    parser.add_argument("-outdir", "-o", help="directory to store histograms in", default=None)
    parser.add_argument("-width", "-w", help="width of bin in MHz", type=float, default=1)
    parser.add_argument("-save", "-s", help="save all histogram data in a csv file", action="store_true")
    parser.add_argument("-notch_filter", "-nf", help="exclude data that was collected within GBT's notch filter", action="store_true")
    args = parser.parse_args()

    print("Gathering files...", end="")
    dat_files = glob.glob(args.folder+"/*.dat")
    print("Done.")

    print("removing DC spikes...", end="")
    dat_files = so.remove_spikes(dat_files, args.band)
    print("Done.")

    # make the first histogram
    print("Calculating first histogram...", end="")
    file_name = os.path.basename(dat_files[0])
    total_hist, bin_edges = so.calculate_hist(dat_files[0], args.band, bin_width=args.width)
    print("Done.")

    # add the individual histogram to a DataFrame
    all_histograms_df = pd.DataFrame()
    temp_dict = {"filename":file_name}
    for i in range(len(total_hist)):
        temp_dict[bin_edges[i]] = total_hist[i]
    all_histograms_df = all_histograms_df.append(temp_dict, ignore_index=True)

    # make the rest of the histograms
    print("Calculating remaining histograms...", end="")
    for i in trange(len(dat_files)-1):
        file_name = os.path.basename(dat_files[i+1])
        hist, edges = so.calculate_hist(dat_files[i+1], args.band, bin_width=args.width)
        temp_df = pd.DataFrame()
        temp_dict = {"filename":file_name}
        for j in range(len(total_hist)):
            temp_dict[bin_edges[j]] = hist[j]
        temp_df = temp_df.append(temp_dict, ignore_index=True)
        all_histograms_df = all_histograms_df.append(temp_df, ignore_index=True)
        total_hist += hist
    print("Done.")

    # restructure DataFrame
    keep_keys = ["filename"]
    for i in range(len(bin_edges[:-1])):
        keep_keys.append(bin_edges[i])
    all_histograms_df = all_histograms_df[keep_keys]

    # store the histogram in a DataFrame
    data_dict = {"frequency":edges[:-1], "count":total_hist}
    df = pd.DataFrame(data_dict)

    # remove notch filter frequencies
    if args.band=="L":
        if args.notch_filter:
            print("Excluding hits in the range 1200-1341 MHz")
            df = df[(df["frequency"] < 1200) | (df["frequency"] > 1341)]
    
    if args.band=="S":
        if args.notch_filter:
            print("Excluding hits in the range 2300-2360 MHz")
            df = df[(df["frequency"] < 2300) | (df["frequency"] > 2360)]

    # check if including notch filter
    if not args.notch_filter:
        if args.band == "L" or args.band == "S":
            filter_flag = "_with_notch_data"
        else:
            filter_flag = ""
    else:
        filter_flag = ""

    if args.outdir is not None:
        outdir = args.outdir+"/"
    else:
        outdir = ""

    # save histogram plot
    plt.figure(figsize=(20, 10))
    plt.bar(df["frequency"], df["count"], width=1)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Count")
    plt.title("%s Band turboSETI Histogram with n=%s files"%(args.band, len(dat_files)))
    plt.savefig("%s%s_band_turboSETI_hist.pdf"%(outdir, args.band))

    # save histogram grid DataFrame as a csv
    if args.save:
        all_histograms_df.to_csv("%s%s_band_ALL_turboSETI_hist_%s.csv"%(outdir, args.band, filter_flag), index=False)
        df.to_csv("%s%s_band_turboSETI_hist.csv"%(outdir, args.band))