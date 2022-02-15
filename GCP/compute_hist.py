import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from tqdm import trange
import argparse
from . import spectral_occupancy as so
from . import check_nodes as cn

def calculate_hist(tbl, GBT_band, bin_width=1, threshold=2048, check_dropped=False, filename=None, dropped_df=None):
    """
    calculates a histogram of the number of hits for a single .dat file
    
    Arguments
    ----------
    tbl : pandas.core.frame.DataFrame
        DataFrame containing the data from running the 
        Energy Detection algorithm on an observation
    GBT_band : str
        the band at which the data was collected
        choose from {"L", "S", "C", "X"}
    bin_width : float
        width of the hisrogram bins in units of MHz
        The default is 1 Mhz
    threshold : float
        The minimum statistic for which data will be included
        The default is 2048
    check_dropped : bool
        flag to determine whether or not to exclude regions
        of the spectrum that come from dropped compute nodes
    filename : str
        name of the source file that the data comes from. This
        is used to look up whether the file is missing nodes
    dropped_df : pandas.core.frame.DataFrame
        the DataFrame containing information about the 
        files with dropped nodes
        
    Returns
    --------
    hist : numpy.ndarray 
        the count of hits in each bin
    bin_edges : numpy.ndarray
        the edge values of each bin. Has length Nbins+1
    """
    
    # reduce the data
    tbl = tbl[tbl["statistic"] >= threshold]

    # make the bins of the histogram
    # band boundaries as listed in Traas 2021
    if GBT_band=="L":
        min_freq = 1100
        max_freq = 1901
    if GBT_band=="S":
        min_freq = 1800
        max_freq = 2801
    if GBT_band=="C":  
        min_freq = 4000
        max_freq = 7801
    if GBT_band=="X":
        min_freq = 7800
        max_freq = 11201

    bins = np.arange(min_freq, max_freq+0.5*bin_width, bin_width)#np.linspace(min_freq, max_freq, int((max_freq - min_freq)/bin_width), endpoint=True)
    hist, bin_edges = np.histogram(tbl["freqs"], bins=bins)
    if check_dropped:
        hist = check_drops(filename, GBT_band, dropped_df, hist, bin_edges)
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

def remove_DC_spikes(df, header_path):
    header = read_header(header_path)
    fch1, foff, nfpc, num_course_channels = grab_parameters(header)
    spike_channels_list = spike_channels(num_course_channels, nfpc)
    freqs_fine_channels_list = freqs_fine_channels(spike_channels_list, fch1, foff)

    freqs = df["freqs"]
    keep_mask = np.in1d(freqs, freqs_fine_channels_list, invert=True)
    return df[keep_mask]

def check_drops(filename, GBT_band, dropped_df, hist, bin_edges):
    """
    Arguments
    ----------
    filename : str
        name of the source file that the data comes from. This
        is used to look up whether the file is missing nodes
    GBT_band : str
        the band at which the data was collected
        choose from {"L", "S", "C", "X"}
    dropped_df : pandas.core.frame.DataFrame
        the DataFrame containing information about the 
        files with dropped nodes
    hist : numpy.ndarray 
        the original count of hits in each bin
    bin_edges : numpy.ndarray
        the edge values of each bin. Has length Nbins+1
    
    
    Returns
    --------
    hist : numpy.ndarray 
        the count of hits in each bin after correcting for 
        dropped compute nodes
    """
    this_file = filename.split("/")[-1]
    drop_files = dropped_df["filename"].values
    drop_flag = False
    for i in range(len(drop_files)):
        short_name = drop_files[i].split(".")[0]
        if short_name == this_file:
            drop_flag = True
            drop_index = i
            break
    if drop_flag:
        missing_row = dropped_df.iloc[drop_index]
        dropped_nodes = missing_row["dropped node"].split(" ")
        boundary_dict = cn.node_boundaries(GBT_band, output="dict")
        for node in dropped_nodes:
            upper_bound = boundary_dict[node]
            lower_bound = boundary_dict[node] - 188
            temp_bounds = bin_edges[:-1]
            mask = np.where((temp_bounds <= upper_bound) & (temp_bounds >= lower_bound))
            hist[mask] = 0
    return hist

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generates a histogram from csv files outputted from the energy detection algorithm")
    parser.add_argument("band", help="the GBT band that the data was collected from. Either L, S, C, or X")
    parser.add_argument("folder", help="folder where the csv files are stored")
    parser.add_argument("-width", "-w", help="width of bin in MHz, default is 1 MHz", default=1)
    parser.add_argument("-threshold", "-t", help="minimum statistic value that will be preserved, default is 2048", default=2048)
    parser.add_argument("-outdir", "-o", help="directory to store histograms in", default=None)
    parser.add_argument("-notch_filter", "-nf", help="exclude data that was collected within GBT's notch filter when generating the plot", action="store_true")
    parser.add_argument("-save", "-s", help="save histogram data in a csv file", action="store_true")
    parser.add_argument("-check_drops", "-c", help="filepath to the csv file summarizing dropped nodes")
    args = parser.parse_args()

    # gather the folders
    print("Gathering folders...", end="")
    folder_paths = glob.glob(args.folder+"/*")
    print("Done gathering %s folders"%len(folder_paths))
    csv_name = "/all_info_df.csv"
    pickle_name = "/header.pkl"

    # make the first histogram
    print("Calculating first histogram...", end="")
    if args.check_drops:
        dropped_df = pd.read_csv(args.check_drops)
    else:
        dropped_df = None
    first_file = pd.read_csv(folder_paths[0]+csv_name)
    file_name = folder_paths[0].split("/")[-1]
    total_hist, bin_edges = calculate_hist(first_file, args.band, bin_width=args.width, threshold=float(args.threshold), 
                                            check_dropped=args.check_drops, filename=file_name, dropped_df=dropped_df)
    print("Done.")

    # add the individual histogram to a DataFrame
    all_histograms_df = pd.DataFrame()
    temp_dict = {"filename":folder_paths[0].split("/")[-1]}
    for i in range(len(total_hist)):
        temp_dict[bin_edges[i]] = total_hist[i]
    all_histograms_df = all_histograms_df.append(temp_dict, ignore_index=True)

    # make the rest of the histograms
    print("Calculating remaining histograms...", end="")
    for i in trange(len(folder_paths)-1):
        this_file = pd.read_csv(folder_paths[i+1]+csv_name)
        file_name = folder_paths[i+1].split("/")[-1]
        hist, edges = calculate_hist(this_file, args.band, bin_width=args.width, threshold=float(args.threshold),
                                      check_dropped=args.check_drops, filename=file_name, dropped_df=dropped_df)
        temp_df = pd.DataFrame()
        temp_dict = {"filename":folder_paths[i+1].split("/")[-1]}
        for i in range(len(total_hist)):
            temp_dict[bin_edges[i]] = hist[i]
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

    # remove notch fileter frequencies 
    if args.band=="L":
        if args.notch_filter:
            print("Excluding hits in the range 1200-1341 MHz")
            df = df[(df["frequency"] < 1200) | (df["frequency"] > 1341)]
            column_names = all_histograms_df.columns[1:]
            all_freq_keys = column_names.astype(float)
            mask = np.where((all_freq_keys < 1200) | (all_freq_keys > 1341))
            keep_keys = all_freq_keys[mask]
            all_histograms_df = all_histograms_df[ ["filename", *list(keep_keys)]]

    if args.band=="S":
        if args.notch_filter:
            print("Excluding hits in the range 2300-2360 MHz")
            df = df[(df["frequency"] < 2300) | (df["frequency"] > 2360)]
            column_names = all_histograms_df.columns[1:]
            all_freq_keys = column_names.astype(float)
            mask = np.where((all_freq_keys < 2300) | (all_freq_keys > 2360))
            keep_keys = all_freq_keys[mask]
            all_histograms_df = all_histograms_df[ ["filename", *list(keep_keys)]]
    
    # check if including notch filter
    if not args.notch_filter:
        if args.band == "L" or args.band == "S":
            filter_flag = "_with_notch_data"
        else:
            filter_flag = ""
    else:
        filter_flag = ""

    # save histogram plot
    plt.figure(figsize=(20, 10))
    plt.bar(df["frequency"], df["count"], width=1)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Count")
    plt.title("%s Band Energy Detection Histogram with n=%s files and threshold=%s"%(args.band, len(folder_paths), args.threshold))
    if args.outdir is not None:
        outdir = args.outdir+"/"
        plt.savefig("%s%s_band_energy_detection_hist_threshold_%s%s.pdf"%(outdir, args.band, args.threshold, filter_flag))
    else:
        plt.savefig("%s_band_energy_detection_hist_threshold_%s%s.pdf"%(args.band, args.threshold, filter_flag))

    # save histogram DataFrame as a csv
    if args.save:
        if args.outdir is not None:
            outdir = args.outdir+"/"
        else:
            outdir = ""
        df.to_csv("%s%s_band_energy_detection_hist_threshold_%s%s.csv"%(outdir, args.band, args.threshold, filter_flag), index=False)
        all_histograms_df.to_csv("%s%s_band_ALL_energy_detection_hist_threshold_%s%s.csv"%(outdir, args.band, args.threshold, filter_flag), index=False)
    print("ALL DONE.")
