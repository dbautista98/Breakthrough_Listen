import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import turbo_seti.find_event as find
import glob
import argparse
import os
from tqdm import trange
import pickle

def calculate_hist(csv_file, header_file, GBT_band, bin_width=1, threshold=2048):
    """
    calculates a histogram of the number of hits for a single .dat file
    
    Arguments
    ----------
    csv_file : str
        filepath to the .csv file
    header_file : str
        filepath to the .pkl file containing the header
    GBT_band : str
        the band at which the data was collected
        choose from {"L", "S", "C", "X"}
    bin_width : float
        width of the hisrogram bins in units of MHz
        The default is 1 Mhz
    threshold : float
        The minimum statistic for which data will be included
        The default is 2048
        
    Returns
    --------
    hist : numpy.ndarray 
        the count of hits in each bin
    bin_edges : numpy.ndarray
        the edge values of each bin. Has length Nbins+1
    """
    # read in the data
    tbl = pd.read_csv(csv_file)

    # remove DC spikes
    tbl = remove_DC_spikes(tbl, header_file)

    # filter above threshold
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

def calculate_proportion(file_list, header_list, GBT_band, notch_filter=False, bin_width=1, threshold=2048):
    """
    Takes in a list of .dat files and makes a true/false table of hits in a frequency bin
    
    Arguments
    ----------
    file_list : list
        A python list containing the filepaths to .csv 
        files that will be used to calculate the 
        spcetral occupancy
    header_list : list
        A python list containing the filepaths to .pkl
        files that will be used to locate DC spike channels
        to remove from data
    GBT_band : str
        the band at which the data was collected
        choose from {"L", "S", "C", "X"}
    notch_filter : bool
        A flag indicating whether or not to remove data 
        that fell within the notch filter. Note to user:
        only L and S band have notch filters
    bin_width : float
        width of the hisrogram bins in MHz
    """
    edges = []
    histograms = []
    min_freq = 0
    max_freq = 1e9
    
    print("Calculating histograms...",end="")
    #calculate histogram for the .dat file and check the boundaries on the data
    for i in trange(len(file_list)):
        hist, bin_edges = calculate_hist(file_list[i], header_list[i], GBT_band, bin_width, threshold)
        if min(bin_edges) > min_freq:
            min_freq = min(bin_edges)
        if max(bin_edges) < max_freq:
            max_freq = max(bin_edges)
        edges.append(bin_edges)
        histograms.append(hist)
    print("Done.")  
    
    #create the dataframe and add the frequency bins to column 0
    df = pd.DataFrame()
    df.insert(0, "freq", edges[0][:-1])
    
    #check if there is a hit in the frequency bin and insert value to dataframe
    for i in range(len(histograms)):
        colname = "file"+str(i)
        found_hit = histograms[i] > 0
        df.insert(len(df.columns), colname, found_hit.astype(int))
    
    #exclude entries in the GBT data due to the notch filter exclusion
    bin_edges = np.linspace(min_freq, max_freq, int((max_freq-min_freq)/bin_width), endpoint=True)
    if GBT_band=="L":
        if notch_filter:
            print("Excluding hits in the range 1200-1341 MHz")
            df = df[(df["freq"] < 1200) | (df["freq"] > 1341)]
            first_edge = np.arange(min_freq, 1200, bin_width)
            second_edge= np.arange(1341, max_freq, bin_width) 
            bin_edges = np.append(first_edge, second_edge)
    
    if GBT_band=="S":
        if notch_filter:
            print("Excluding hits in the range 2300-2360 MHz")
            df = df[(df["freq"] < 2300) | (df["freq"] > 2360)]
            first_edge = np.arange(min_freq, 2300, bin_width)
            second_edge= np.arange(2360, max_freq, bin_width) #may or may not need max_freq+1
            bin_edges = np.append(first_edge, second_edge)

     
    # sum up the number of entries that have a hit and divide by the number of .dat files
    data_labels = df.columns[2:]
    total = df["file0"].values
    for label in data_labels:
        total = total + df[label].values
    
    return bin_edges, total/len(file_list) 

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
    parser = argparse.ArgumentParser(description="generates a histogram of the spectral occupancy from a given the output of the energy detection algorithm")
    parser.add_argument("band", help="the GBT band that the data was collected from. Either L, S, C, or X")
    parser.add_argument("folder", help="directory containing the folders which store the energy detection output")
    parser.add_argument("-width", "-w", help="width of bin in Mhz", type=float, default=1)
    parser.add_argument("-notch_filter", "-nf", help="exclude data that was collected within GBT's notch filter when generating the plot", action="store_true")
    parser.add_argument("-threshold", "-t", help="threshold below which all hits will be excluded. Default is 4096", type=float, default=4096)
    parser.add_argument("-save", "-s", help="save the histogram bin edges and heights", action="store_true")
    args = parser.parse_args()

    print("Gathering files...", end="")
    files = glob.glob(args.folder+"/*")
    headers = []
    for i in range(len(files)):
        headers.append(files[i] + "/header.pkl")
        files[i] = files[i] + "/all_info_df.csv"
    print("Done.")


    bin_edges, prob_hist = calculate_proportion(files, headers, bin_width=args.width, GBT_band=args.band, notch_filter=args.notch_filter, threshold=args.threshold)

    if args.save:
        print("Saving histogram data")
        to_save = {"bin_edges":bin_edges, "bin_heights":prob_hist, "band":args.band, "threshold":args.threshold, "bin width":args.width, "algorithm":"energy detection", "n files":len(files)}
        filename = "energy_detection_%s_band_spectral_occupancy_%s_MHz_bins_%s_threshold.pkl"%(args.band, args.width, args.threshold)
        with open(filename, "wb") as f:
            pickle.dump(to_save, f)

    print("Saving plot...",end="")
    plt.figure(figsize=(20, 10))
    plt.bar(bin_edges[:-1], prob_hist, width = .99) 
    plt.xlabel("Frequency [Mhz]")
    plt.ylabel("Fraction with Hits")
    plt.title("Spectral Occupancy with threshold of %s: n=%s"%(args.threshold, len(files)))
    plt.savefig("%s_band_spectral_occupancy_thresh_%s_bin_%sMHz.pdf"%(args.band, int(args.threshold), args.width))
    print("Done")
