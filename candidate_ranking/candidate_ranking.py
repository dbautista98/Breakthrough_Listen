import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import turbo_seti.find_event as find
import glob
import argparse
import os
import pickle
from tqdm import trange
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time

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
    print("reading data")
    for i in trange(len(hit_list)):
        temp_df = pd.read_csv(hit_list[i])
        temp_df["event_csv_path"] = hit_list[i]
        df = df.append(temp_df, ignore_index=True)
    return df

def rank_candidates(hit_freq, occupancy_dict):
    bin_edges = occupancy_dict["bin_edges"][:-1]
    occupancy = occupancy_dict["bin_heights"]
    rankings = np.empty_like(hit_freq)
    print("calculating rankings")
    for i in trange(len(hit_freq)):
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

def make_plots(band_df, band_rankings, band, save_dir, bins=50, grid_sky=False):
    if grid_sky:
        grid_str = " grid sky"
    else:
        grid_str = " all sky"

    plt.figure(figsize=(10,5))
    _ = plt.hist(band_rankings, bins=bins)
    plt.xlabel("ranking")
    plt.ylabel("count")
    plt.title("%s-band Rankings\nN = %s candidate hits\nn = %s unique events"%(band, len(band_rankings), len(np.unique(band_df["Hit_ID"]))))
    plt.xlim([-0.05,1.05])
    plt.savefig(save_dir + "/%sband_ranking%s.png"%(band.lower(), grid_str.replace(" ", "_")), bbox_inches="tight", transparent=False)
    plt.savefig(save_dir + "/%sband_ranking%s.pdf"%(band.lower(), grid_str.replace(" ", "_")), bbox_inches="tight", transparent=False)

def notch_filter(band, occupancy_dict):
    if band == "C" or band == "X":
        pass
    elif band == "L":
        notch_region = np.where((occupancy_dict["bin_edges"][:-1] > 1200) & (occupancy_dict["bin_edges"][:-1] < 1341))
        occupancy_dict["bin_heights"][notch_region] = 1
    elif band == "S":
        notch_region = np.where((occupancy_dict["bin_edges"][:-1] > 2300) & (occupancy_dict["bin_edges"][:-1] < 2360))
        occupancy_dict["bin_heights"][notch_region] = 1
    else:
        raise Exception("invalid band")
    return occupancy_dict

def make_scatter(band_df, band_rankings, band, save_dir, grid_sky=False):
    if grid_sky:
        grid_str = " grid sky"
    else:
        grid_str = " all sky"

    plt.figure(figsize=(10,5))
    plt.scatter(band_df["Freq"].values, band_rankings, s=2)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("ranking")
    if band == "L":
        plt.axvspan(1200, 1341, alpha=0.5, color='red', label="notch filter region")
        plt.legend()
    if band == "S":
        plt.axvspan(2300, 2360, alpha=0.5, color='red', label="notch filter region")
        plt.legend()
    plt.title("%s-band ranking vs frequency%s\nN = %s candidate hits\nn = %s unique events"%(band, grid_str, len(band_rankings), len(np.unique(band_df["Hit_ID"]))))
    plt.ylim([-0.05,1.05])
    plt.grid()
    plt.savefig(save_dir + "/%sband_freq_vs_ranking%s.png"%(band.lower(), grid_str.replace(" ", "_")), bbox_inches="tight", transparent=False)
    plt.savefig(save_dir + "/%sband_freq_vs_ranking%s.pdf"%(band.lower(), grid_str.replace(" ", "_")), bbox_inches="tight", transparent=False)

def freq_mask(eventdf, band):
    """
    This is the same mask that Carmen and Noah have been 
    using to cut the data to only include the "good" regions
    of the spectrum
    """
    efreqs = pd.to_numeric(eventdf.Freq)

    if band == "L":
        eventmask = (efreqs/1000>1.10) & (efreqs/1000<1.90) &  ((efreqs/1000 < 1.20) | (efreqs/1000 > 1.34))
    elif band == "S":
        eventmask = ((efreqs/1000>1.80) & (efreqs/1000<2.80)) & ((efreqs/1000 < 2.30) | (efreqs/1000 > 2.36))
    elif band == "C":
        eventmask = ((efreqs/1000>4.00) & (efreqs/1000<7.80))
    elif band == "X":
        eventmask = ((efreqs/1000>7.80) & (efreqs/1000<11.20))
    else:
        raise Exception("invalid band")

    eventdf = eventdf[eventmask]

    return eventdf

def get_AltAz(df):
    """
    Takes a dataframe containing the RA, DEC and MJD values 
    and returns the altitude and azimuth angles the telescope
    was pointing during the observation

    Arguments
    ----------
    df : pandas.core.frame.DataFrame
        DataFrame containing right ascension (RA), declinaiton (DEC)
        and modified julian date (MJD) values for each observation

    Returns
    --------
    ALT : astropy.coordinates.angles.Latitude
        The altidude angle of the telescope; how many degrees above the horizon
        the telescope was pointing
    AZ : astropy.coordinates.angles.Longitude
        the azimuthal angle the telescope was pointing; how many degrees from 
        true north it was aimed, following the right hand rule
    """
    targets = SkyCoord(df["RA"].values, df["DEC"].values, unit=(u.hourangle, u.deg), frame="icrs")          
    times = Time(np.array(df["MJD"].values, dtype=float), format="mjd") 
    gbt = EarthLocation(lat=38.4*u.deg, lon=-79.8*u.deg, height=808*u.m)
    gbt_altaz_transformer = AltAz(obstime=times, location=gbt)
    gbt_target_altaz = targets.transform_to(gbt_altaz_transformer)
    ALT = gbt_target_altaz.alt
    AZ = gbt_target_altaz.az

    return ALT, AZ

def all_sky_rank(df, band, outdir="./"):
    # grab spectral occupancy data paths and sort bands
    spectral_occupancy = glob.glob("all_sky/*.pkl")
    spectral_occupancy.sort()

    if len(spectral_occupancy) == 0:
        raise Exception("no pickles")

    # read the spectral occupancy data into memory
    if band.upper() == "L":
        output = read_pickle(spectral_occupancy[1])
        output = notch_filter("L", output)
    elif band.upper() == "S":
        output = read_pickle(spectral_occupancy[2])
        output = notch_filter("S", output)
    elif band.upper() == "C":
        output = read_pickle(spectral_occupancy[0])
    elif band.upper() == "X":
        output = read_pickle(spectral_occupancy[3])
    else:
        print("Please enter a valid band. Choose from {L, S, C, X}")
        exit()

    # rank candidates and save plots, csv
    band_rankings = rank_candidates(band_df["Freq"].values, output)

    make_plots(band_df=band_df, band_rankings=band_rankings, band=args.band, save_dir=outdir)
    make_scatter(band_df=band_df, band_rankings=band_rankings, band=args.band, save_dir=outdir)

    band_df.insert(0, "ranking", band_rankings)
    band_df.to_csv(outdir + "/%s_band_candidate_rankings_all_sky.csv"%band, index=False)

def grid_rank(df, band, outdir="./"):
    alt_bins = np.linspace(0,90,num=4, endpoint=True)
    az_bins = np.linspace(0,360, num=5, endpoint=True)

    alt, az = get_AltAz(df)
    df["ALT"] = alt
    df["AZ"] = az

    rankings_df = pd.DataFrame()

    for i in range(len(alt_bins)-1):
        for j in range(len(az_bins)-1):
            mask = np.where((df["ALT"] >= alt_bins[i]) & (df["ALT"] < alt_bins[i+1]) & (df["AZ"] >= az_bins[j]) & (df["AZ"] < az_bins[j+1]))
            this_df = df.iloc[mask]
            spectral_occupancy_pickle = glob.glob("grid_sky/*%s_band*alt-(%s,%s)_az-(%s,%s)*pkl"%(band, alt_bins[i], alt_bins[i+1], az_bins[j], az_bins[j+1]))
            if len(spectral_occupancy_pickle) == 0:
                raise Exception("no pkl file found")
            output = read_pickle(spectral_occupancy_pickle[0])
            output = notch_filter(band, output)
            this_rankings = rank_candidates(this_df["Freq"].values, output)
            this_df.insert(0, "ranking", this_rankings)
            rankings_df = rankings_df.append(this_df, ignore_index=True)

    make_plots(band_df=rankings_df, band_rankings=rankings_df["ranking"].values, band=args.band, save_dir=outdir, grid_sky=True)
    make_scatter(band_df=rankings_df, band_rankings=rankings_df["ranking"].values, band=args.band, save_dir=outdir, grid_sky=True)

    rankings_df.to_csv(outdir + "/%s_band_candidate_rankings_grid_sky.csv"%band, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="calculates the weight of interest in a candidate signal that was detected by turboSETI's find event pipeline, note that the present form of the program should be run on a single band's data at a time")
    parser.add_argument("band", help="the GBT band that the data was collected from. Either L, S, C, or X")
    parser.add_argument("-folder", "-f", help="directory .csv files are held in")
    parser.add_argument("-text", "-t", help="a .txt file to read the filepaths of the .csv files", action=None)
    parser.add_argument("-outdir", '-o', help="directory to output results to", default=os.getcwd())
    parser.add_argument("-grid_ranking", "-g", help="rank candidates based on the region of sky they come from. groups candidates together with other hits from same patch of sky", default=False, action="store_true")
    args = parser.parse_args()

    # grab turboSETI cadence search results 
    if args.folder is not None:
        paths = glob.glob(args.folder + "/*csv")
        band_df = combine_frames(paths)
    elif args.text is not None:
        paths = read_txt(args.text)
        band_df = combine_frames(paths)
    else:
        print("please enter either a directory path or a txt file containing the paths to the files")
        exit()

    # clean spectrum to only include wanted regions
    band_df = freq_mask(band_df, args.band)

    if not args.grid_ranking:
        all_sky_rank(band_df, args.band, args.outdir)
    else:
        grid_rank(band_df, args.band, args.outdir)
