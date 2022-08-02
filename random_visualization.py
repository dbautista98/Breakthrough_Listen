import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse 
from tqdm import trange

try:
    import spectral_occupancy as so
except:
    from . import spectral_occupancy as so

def reduce_frames(csv_list):
    df = pd.DataFrame()
    for i in trange(len(csv_list)):
        temp = pd.read_csv(csv_list[i])
        df = df.append(temp, ignore_index=True)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generates a histogram of the spectral occupancy from a given set of .dat files")
    parser.add_argument("band", help="the GBT band that the data was collected from. Either L, S, C, or X")
    parser.add_argument("csv_dir", help="directory .csv files are held in")
    parser.add_argument("outdir", help="directory where the output cadences searches will be saved")
    args = parser.parse_args()

    csv_paths = glob.glob(args.csv_dir + "/*/*.csv")
    total_observations = glob.glob(args.csv_dir + "*")
    df = reduce_frames(csv_paths)

    hist, bin_edges = so.calculate_hist(df, args.band)

    # account for triple counting hits due to version of turboSETI used
    hist = hist/3

    plt.figure(figsize=(10,5))
    plt.bar(bin_edges[:-1], hist, width=1)
    plt.xlabel("Frequency [Mhz]")
    plt.ylabel('Number of Candiate "Signals"')
    plt.title("Histogram of Chance Candidates\nN = %s observations and n = %s with candidates"%(len(total_observations), len(csv_paths)))
    plt.savefig(args.outdir + "%s_band_random_chance_candidates.pdf"%args.band, bbox_inches="tight", transparent=False)
    plt.savefig(args.outdir + "%s_band_random_chance_candidates.png"%args.band, bbox_inches="tight", transparent=False)