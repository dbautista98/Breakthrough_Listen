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
    """
    Takes in path to csv file, removes DC spike channels
    and and filters data to be above threshold value
    
    Arguments
    ----------
    csv_path : str
        Filepath to .csv file containing results of running
        Energy Detection on GBT data
    header_path : str
        Filepath to .pkl file containing the header data
        to the corresponding csv file. 
        The .pkl file should be in the same folder as the .csv
    threshold : float
        Minimum statistic value for energy detection data
        A good default value to give spectral occupancy results
        similar to turboSETI results is 4096

    Returns
    --------
    tbl : pandas.core.frame.DataFrame
    """
    # read in data
    tbl = pd.read_csv(csv_path)

    # remove DC spikes
    tbl = so.remove_DC_spikes(tbl, header_path)

    # filter above threshold
    tbl = tbl.iloc[np.where(tbl["statistic"] > threshold)]
    return tbl

def split_data(tbl, start, stop):
    """
    Returns a subset of the DataFrame with frequencies
    between the specified boundaries

    Arguments
    ----------
    tbl : pandas.core.frame.DataFrame
        A pandas DataFrame with data to be split up by frequency
        Must contain a column labeled "freqs"
    start : float
        lower bound of allowed frequencies
    stop : float
        upper bound of allowed frequencies

    Returns
    --------
    tbl : pandas.core.frame.DataFrame
        A pandas dataframe that contains a smaller 
        inteval of data 
    """
    mask = np.where((tbl["freqs"] >= start) & (tbl["freqs"] < stop))
    return tbl.iloc[mask]

def interval_fraction(tbl, start, stop, fine_channel_width=3/2**20):
    """
    Determines the fraction of fine channels in a given interval 
    that measure a signal. The fraction is determined by creating
    a histogram with bin widths equal to the fine channel width
    and then counting how many bins measure one or more signals

    Arguments
    ----------
    tbl : pandas.core.frame.DataFrame
        DataFrame containing Energy Detection data
    start : float
        lower bound of allowed frequencies
    stop : float
        upper bound of allowed frequencies
    fine_channel_width : float
        width of fine channel in MHz. Default 
        width is 3MHz/2^20 for HSR 
        files per Lebofsky et al 2019

    Returns
    --------
    fraction : float
        The fraction of fine channels for the frequency
        range that detected a signal
    """
    tbl = split_data(tbl, start, stop)
    bins = np.arange(start, stop + 0.5*fine_channel_width, fine_channel_width)#np.linspace(start, stop, int((stop-start)/fine_channel_width)+1, endpoint=True) # the +1 is to make the number of bins correct after constructing histogram
    hist, bin_edges = np.histogram(tbl["freqs"], bins=bins)
    fraction = np.sum((hist > 0))/len(hist)
    return fraction

def one_file(csv_data, GBT_band, bin_width=1, fine_channel_width=3/2**20, notch_filter=False):
    """
    Arguments
    ----------
    csv_data : pandas.core.frame.DataFrame
        DataFrame containing the results of running 
        Energy Detection on GBT dat files
    GBT_band : str
        The band that the data was collected 
        with. Is one of {L, S, C, X}
    bin_width : float
        Width of the bins when finding the Spectral Occupancy
        of the data. Default value is 1
    fine_channel_width : float
        width of fine channel in MHz. Default 
        width is 3MHz/2^20 for HSR 
        files per Lebofsky et al 2019
    notch_filter : bool
        Option to remove the data with frequencies
        covered by GBT's notch filter

    Returns
    --------
    fractions : numpy.ndarray
        Array containing the fraction of fine channels
        that measured a signal in a given step size
    frequencies : numpy.ndarray
        Array containing the starting frequency of
        each bin, in units of MHz
    """
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
        bin_fractions[i] = interval_fraction(csv_data, spectral_occupancy_bins[i], spectral_occupancy_bins[i] + bin_width, fine_channel_width=fine_channel_width)
    
    # store data in dataframe to remove notch filter
    data = {"freq":spectral_occupancy_bins, "fractions":bin_fractions}
    df = pd.DataFrame(data)

    # remove notch filters
    if GBT_band=="L":
        if notch_filter:
            df = df[(df["freq"] < 1200) | (df["freq"] > 1341)]
    
    if GBT_band=="S":
        if notch_filter:
            df = df[(df["freq"] < 2300) | (df["freq"] > 2360)]
    
    return df["fractions"].values, df["freq"].values


if __name__ == "__main__":
    default_fine_channel_width = 3/(2**20) # Lebofsky et al 2019 for HSR (high spectral resolution) files 
    parser = argparse.ArgumentParser(description="determines the fraction of fine channels that have hits in a spectral occupancy bin")
    parser.add_argument("band", help="the GBT band that the data was collected from. Either L, S, C, or X")
    parser.add_argument("folder", help="directory containing the folders which store the energy detection output")
    parser.add_argument("-width", "-w", help="width of the spectral occupancy bin in Mhz. Default is 1MHz", type=float, default=1)
    parser.add_argument("-threshold", "-t", help="threshold below which all hits will be excluded. Default is 4096", type=float, default=4096)
    parser.add_argument("-notch_filter", "-nf", help="exclude data that was collected within GBT's notch filter when generating the plot", action="store_true")
    parser.add_argument("-fine", "-f", help="width of fine channel in MHz. Default width is 3MHz/2^20 for HSR files per Lebofsky et al 2019", type=float, default=default_fine_channel_width)
    args = parser.parse_args()

    files = glob.glob(args.folder+"/*")
    dataset_fractions = 0
    n_files = len(files)
    filenames, file_fractions = [], []
    print("Processing files...")
    for i in trange(len(files)):
        data = (files[i]+"/all_info_df.csv")
        header = (files[i] + "/header.pkl")
        file_name = (files[i].split("/")[-1])
        tbl = read_data(data, header, threshold=args.threshold)
        bin_fractions, frequency_bins = one_file(tbl, args.band, bin_width=args.width, fine_channel_width=args.fine, notch_filter=args.notch_filter)
        dataset_fractions += bin_fractions
        filenames.append(file_name)
        file_fractions.append(bin_fractions)

    # normalize data by number of files
    dataset_fractions = dataset_fractions/n_files

    # make dataframe of fraction data for all files
    file_fractions = np.array(file_fractions)
    df = pd.DataFrame(data=file_fractions.T, index=frequency_bins, columns=filenames)

    print("Saving fine channel data")
    to_save = {"fine channel fraction":dataset_fractions, "frequency bin":frequency_bins, "dataframe":df, "bin_width":args.width, "fine_channel_width":args.fine, "band":args.band, "threshold":args.threshold, "algorithm":"energy detection", "n files":len(files)}
    filename = "energy_detection_%s_band_fine_channel_fraction.pkl"%(args.band)
    with open(filename, "wb") as f:
        pickle.dump(to_save, f)

    print("Done!")