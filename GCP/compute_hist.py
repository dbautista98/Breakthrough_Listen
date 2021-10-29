import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from tqdm import trange
import argparse

def calculate_hist(csv_file, GBT_band, bin_width=1, threshold=2048, notch_filter=False):
    """
    calculates a histogram of the number of hits for a single .dat file
    
    Arguments
    ----------
    csv_file : str
        filepath to the .csv file
    GBT_band : str
        the band at which the data was collected
        choose from {"L", "S", "C", "X"}
    bin_width : float
        width of the hisrogram bins in units of MHz
        The default is 1 Mhz
    threshold : float
        The minimum statistic for which data will be included
        The default is 2048
    notch_filter : bool
        A flag indicating whether or not to remove data 
        that fell within the notch filter. Note to user:
        only L and S band have notch filters
        
    Returns
    --------
    hist : numpy.ndarray 
        the count of hits in each bin
    bin_edges : numpy.ndarray
        the edge values of each bin. Has length Nbins+1
    """
    
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

    #remove notch filters
    if GBT_band=="L":
        if notch_filter:
            print("Excluding hits in the range 1200-1341 MHz")
            tbl = tbl[(tbl["freqs"] < 1200) | (tbl["freqs"] > 1341)]
    
    if GBT_band=="S":
        if notch_filter:
            print("Excluding hits in the range 2300-2360 MHz")
            tbl = tbl[(tbl["freqs"] < 2300) | (tbl["freqs"] > 2360)]
    
    bins = np.linspace(min_freq, max_freq, int((max_freq - min_freq)/bin_width), endpoint=True)
    hist, bin_edges = np.histogram(tbl["freqs"], bins=bins)
    del tbl
    return hist, bin_edges

def read_header(header_path):
    """
    Takes in a path to pickled header and returns a dictionary 
    containing the header info

    Arguments:
    -----------
    header_path : str
        file path to the location of the .pkl file
    
    Returns:
    --------
    header : dict
        the header info from the corresponding 
        .h5 or fil file
    """
    return pd.read_pickle(header_path)

def spike_channels(num_course_channels, nfpc):
    """makes a spike channels list given a list of channels"""
    spike_channels_list=[]
    for i in np.arange(num_course_channels): 
        spike_channel=(nfpc/2.0)+(nfpc*i)
        spike_channels_list.append(spike_channel)
    return np.asarray(spike_channels_list)

def freqs_fine_channels(spike_channels_list, fch1, foff):
    freqs_fine_channels_list=[]
    for index, value in enumerate(spike_channels_list):
        freq_fine_channel=fch1+foff*value
        if freq_fine_channel>0:
            freq_fine_channel=round(freq_fine_channel, 6)
            freqs_fine_channels_list.append(freq_fine_channel)
        else:
            break
    return np.asarray(freqs_fine_channels_list)

def grab_parameters(header):
    fch1 = header["fch1"]
    foff = header["foff"]
    nfpc=(1500.0/512.0)/abs(foff)
    num_course_channels = header["nchans"]/nfpc
    return fch1, foff, nfpc, num_course_channels

def remove_DC_spikes(df, header_path):#freqs_fine_channels_list, foff):
    header = read_header(header_path)
    fch1, foff, nfpc, num_course_channels = grab_parameters(header)
    spike_channels_list = spike_channels(num_course_channels, nfpc)
    freqs_fine_channels_list = freqs_fine_channels(spike_channels_list, fch1, foff)

    freqs = df["freqs"]
    keep_mask = np.in1d(freqs, freqs_fine_channels_list, invert=True)
    return df[keep_mask]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generates a histogram from csv files outputted from the energy detection algorithm")
    parser.add_argument("band", help="the GBT band that the data was collected from. Either L, S, C, or X")
    parser.add_argument("folder", help="folder where the csv files are stored")
    parser.add_argument("-width", "-w", help="width of bin in MHz, default is 1 MHz", default=1)
    parser.add_argument("-threshold", "-t", help="minimum statistic value that will be preserved, default is 2048", default=2048)
    parser.add_argument("-outdir", "-o", help="directory to store histograms in", default=None)
    parser.add_argument("-notch_filter", "-nf", help="exclude data that was collected within GBT's notch filter when generating the plot", action="store_true")
    args = parser.parse_args()

    print("Gathering folders...", end="")
    folder_paths = glob.glob(args.folder+"/*")
    print("Done gathering %s folders"%len(folder_paths))

    csv_name = "/all_info_df.csv"
    print("Calculating first histogram...", end="")
    total_hist, bin_edges = calculate_hist(folder_paths[0]+csv_name, args.band, bin_width=args.width, threshold=float(args.threshold), notch_filter=args.notch_filter)
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
