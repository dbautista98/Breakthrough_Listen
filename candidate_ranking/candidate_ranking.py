import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import turbo_seti.find_event as find
import glob
import argparse
import os
import pickle

def read_txt(text_file):
    """
    reads a text file with one filepath per
    line and returns a python list where
    each entry is a filepath
    
    Arguments
    ----------
    text_file : str
        A string indicating the location of the 
        text file pointing to the dat files 
    """
    with open(text_file) as open_file:
        lines = open_file.readlines()
    
    for i in range(len(lines)):
        lines[i] = lines[i].replace("\n", "")
    return lines

def read_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def calculate_weight(spectral_occupancy):
    return (1 - spectral_occupancy)**3

def combine_frames(hit_list):
    df = pd.DataFrame()
    for csv in hit_list:
        temp_df = pd.read_csv(csv)
        df = df.append(temp_df, ignore_index=True)
    return df

def rank_candidates(hit_freq, occupancy_dict):
    bin_edges = occupancy_dict["bin_edges"][:-1]
    occupancy = occupancy_dict["bin_heights"]
    rankings = np.empty_like(hit_freq)
    for i in range(len(hit_freq)):
        freq = hit_freq[i]
        hit_index = np.where((bin_edges >= np.floor(freq)) & (bin_edges < np.ceil(freq)))
        hit_occupancy = occupancy[hit_index]
        this_weight = calculate_weight(hit_occupancy)
        # check that the frequency is within the accepted range
        if len(this_weight) > 0: # hit is within the acceptable band range
            rankings[i] = calculate_weight(hit_occupancy)
        else: # hit is outside acceptable band range and is not interesting
            rankings[i] = 0
    return rankings

def make_plots(band_df, occupancy_dict, band, save_dir, bins=50):
    band_rankings = rank_candidates(band_df["Freq"].values, occupancy_dict)
    
    plt.figure()
    _ = plt.hist(band_rankings, bins=bins)
    plt.xlabel("ranking")
    plt.ylabel("count")
    plt.title("%s-band Rankings N = %s candidates"%(band, len(band_rankings)))
    plt.savefig(save_dir + "/%sband_ranking.png"%band.lower(), bbox_inches="tight", transparent=False)
    
    band_df.insert(0, "ranking", band_rankings)
    band_df.to_csv(save_dir + "/%s_band_candidate_rankings.csv"%band, index=False)
    return band_df
    
def make_scatter(band_df, occupancy_dict, band, save_dir):
    band_rankings = rank_candidates(band_df["Freq"].values, occupancy_dict)
    
    plt.figure()
    plt.scatter(band_df["Freq"].values, band_rankings)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("ranking")
    plt.title("%s-band ranking vs frequency N = %s candidates"%(band, len(band_rankings)))
    plt.savefig(save_dir + "/%sband_freq_vs_ranking.png"%band.lower(), bbox_inches="tight", transparent=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="calculates the weight of interest in a candidate signal that was detected by turboSETI's find event pipeline, note that the present form of the program should be run on a single band's data at a time")
    parser.add_argument("band", help="the GBT band that the data was collected from. Either L, S, C, or X")
    parser.add_argument("-folder", "-f", help="directory .dat files are held in")
    parser.add_argument("-text", "-t", help="a .txt file to read the filepaths of the .dat files", action=None)
    parser.add_argument("-outdir", '-o', help="directory to save results output results to", default=os.getcwd())
    args = parser.parse_args()

    # grab spectral occupancy data paths and sort bands
    spectral_occupancy = glob.glob("*pkl")
    spectral_occupancy.sort()

    # grab turboSETI cadence search results 
    if args.folder is not None:
        paths = glob.glob(args.folder + "/*csv")
        cadences = combine_frames(paths)
    elif args.text is not None:
        paths = read_txt(args.text)
        cadences = combine_frames(paths)
    else:
        print("please enter either a directory path or a txt file containign the paths to the files")
        exit()
    
    # read the spectral occupancy data into memory
    if args.band.upper() == "L":
        output = read_pickle(spectral_occupancy[1])
    elif args.band.upper() == "S":
        output = read_pickle(spectral_occupancy[2])
    elif args.band.upper() == "C":
        output = read_pickle(spectral_occupancy[0])
    elif args.band.upper() == "X":
        output = read_pickle(spectral_occupancy[3])
    else:
        print("Please enter a valid band. Choose from {L, S, C, X}")
        exit()

    # rank candidates and save plots, csv
    make_plots(band_df=cadences, occupancy_dict=output, band=args.band, save_dir=args.outdir)
    make_scatter(band_df=cadences, occupancy_dict=output, band=args.band, save_dir=args.outdir)

    
