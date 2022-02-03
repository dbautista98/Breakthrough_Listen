import pandas as pd
import numpy as np
import glob
import argparse
from tqdm import trange
from scipy import stats
import os
import matplotlib.pyplot as plt
import compute_hist as ch

def multi_hist(df, GBT_band, threshold_arr, bin_width=1):
    """
    creates multiple histograms of the energy detection 
    results with different threshold values 

    Arguments
    ----------
    df : pandas.core.frame.DataFrame
        DataFrame containing the data from running the 
        Energy Detection algorithm on an observation
    GBT_band : str
        the band at which the data was collected
        choose from {"L", "S", "C", "X"}
    threshold_arr : list
        The minimum statistic for which data will be included 
        in a histogram
    bin_width : float
        width of the hisrogram bins in units of MHz
        The default is 1 Mhz


    Returns
    --------
    histograms : numpy.ndarray, shape (n_thrsholds, n_bins)
        the count of hits in each bin for each threshold
    bin_edges : numpy.ndarray
        the edge values of each bin. Has length n_bins+1
    """

    # make the first histogram
    first_hist, bin_edges = ch.calculate_hist(tbl=df, GBT_band=GBT_band, bin_width=bin_width, threshold=threshold_arr[0])
    histograms = np.empty(shape=(len(threshold_arr), len(first_hist)))
    histograms[0] = first_hist

    # make the rest of the histograms
    for i in range(1, len(threshold_arr)):
        thresh = threshold_arr[i]
        this_hist, bin_edges = ch.calculate_hist(tbl=df, GBT_band=GBT_band, bin_width=bin_width, threshold=thresh)
        histograms[i] = this_hist
    
    return histograms, bin_edges

def RMSE(true, observed):
    """"
    Calculates the root mean squared error for the given data

    Arguments
    ----------
    true : numpy.ndarray 
        the values we are trying to match
    observed : numpy.ndarray 
        the values we found with a given model
    
    Returns
    --------
    rmse : float
        the root mean squared error of the data
    """
    residuals = true - observed
    if len(observed.shape) > 1:
        return np.sqrt( np.sum(residuals**2, axis=1) / len(residuals))
    return np.sqrt( np.sum(residuals**2) / len(residuals))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compares energy detection histograms under a range of threshold values to the turboSETI histogram")
    parser.add_argument("band", help="the GBT band that the data was collected from. Either L, S, C, or X")
    parser.add_argument("folder", help="folder where the energy detection csv files are stored")
    parser.add_argument("turbo_seti", help="path to the turboSETI histogram that it will be compared to")
    parser.add_argument("-width", "-w", help="width of bin in MHz, default is 1 MHz", default=1)
    parser.add_argument("-outdir", "-o", help="directory to store histograms in", default=None)
    parser.add_argument("-notch_filter", "-nf", help="exclude data that was collected within GBT's notch filter when processing the data", action="store_true")
    parser.add_argument("-save", "-s", help="save histogram data in a csv file", action="store_true")
    parser.add_argument("-plot", "-p", help="plot and save the genereated histograms", action="store_true")
    args = parser.parse_args()


    # establish the threshold range
    thresholds = np.logspace(11, 14, num=50, base=2)

    # collect the folders
    print("Gathering folders...", end="")
    folder_paths = glob.glob(args.folder+"/*")
    print("Done gathering %s folders"%len(folder_paths))

    # make the first set of histograms
    csv_name = "/all_info_df.csv"
    print("Calculating first histogram...", end="")
    first_file = pd.read_csv(folder_paths[0] + csv_name)
    total_hist, bin_edges = multi_hist(first_file, args.band, thresholds, bin_width=args.width)
    print("Done.")

    # make the rest of the histograms
    for i in trange(len(folder_paths)-1):
        this_file = pd.read_csv(folder_paths[i+1]+csv_name)
        hist, edges = multi_hist(this_file, args.band, thresholds, bin_width=args.width)
        total_hist += hist

    # store the histogram data in a dataframe
    data_dict = {"frequency":edges[:-1]}
    for i in range(len(thresholds)):
        data_dict[str(thresholds[i])] = total_hist[i]
    df = pd.DataFrame(data_dict)

    # remove the notch filter data
    if args.band=="L":
        if args.notch_filter:
            print("Excluding hits in the range 1200-1341 MHz")
            df = df[(df["frequency"] < 1200) | (df["frequency"] > 1341)]
    if args.band=="S":
        if args.notch_filter:
            print("Excluding hits in the range 2300-2360 MHz")
            df = df[(df["frequency"] < 2300) | (df["frequency"] > 2360)]

    # save the histograms
    if args.save:
        if args.outdir is not None:
            outdir = args.outdir+"/"
        else:
            outdir = ""
        df.to_csv("%s%s_band_energy_detection_histogram_multiple_thresholds.csv"%(outdir, args.band))
    
    # pick out the threshold keys
    threshold_keys = df.columns[1:]

    # read in the turbo_seti histograms
    turbo_seti = pd.read_csv(args.turbo_seti)
    if len(turbo_seti["frequency"].values) != len(df["frequency"]):
        print("WARNING:  Your data are not the same length. Please double check the inputs")
        print("\tlen(turboSETI):         ", len(turbo_seti["frequency"].values))
        print("\tlen(energy detection):  ", len(df["frequency"]))
    turbo_seti_count = turbo_seti["count"].values

    # calculate the RMSE
    ed = np.array(df[threshold_keys]).T
    ts = turbo_seti_count
    all_rmse = RMSE(ts, ed)
    
    # store results
    results_dict = {"threshold":thresholds, "RMSE":all_rmse}
    results_df = pd.DataFrame(results_dict)
    if args.outdir is not None:
        outdir = args.outdir+"/"
    else:
        outdir = ""
    results_df.to_csv("%s%s_band_multiple_thresholds_RMSE.csv"%(outdir, args.band))

    if args.plot:
        if args.outdir is not None:
            outdir = args.outdir+"/%s_band_histogram_plots/"%args.band
            if not os.path.exists(outdir):
                os.mkdir(outdir)
        else:
            outdir = "%s_band_histogram_plots/"%args.band
            if not os.path.exists(outdir):
                os.mkdir(outdir)
        print("Saving plots...", end="")
        for thresh in threshold_keys:
            print("DEBUG::")
            # print(type(bin_edges))
            print("\tfrequency:       ", (df["frequency"].shape))
            print("\tenergy detection:", (df[thresh].values.shape))
            print("\tturboSETI:       ", (turbo_seti_count.shape))
            plt.figure(figsize=(20,10))
            plt.bar(df["frequency"].values, df[thresh].values, width=1, label="energy detection", alpha=0.5)
            plt.bar(df["frequency"].values, turbo_seti_count, width=1, label="turboSETI", alpha=0.5)
            plt.legend()
            plt.xlabel("Frequency [MHz]")
            plt.ylabel("count")
            plt.title("turboSETI vs Energy Detection (threshold = %s)"%thresh)
            savepath = outdir + "%s_band_turboSETI_vs_energy_detection_histogram_threshold_%s.pdf"%(args.band, thresh)
            plt.savefig(savepath, bbox_inches="tight", transparent=False)
    
    # print the best threshold
    best_threshold = threshold_keys[np.argmin(all_rmse)]
    print("The best threshold value is:", best_threshold)