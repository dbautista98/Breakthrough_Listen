import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse 

try:
    import spectral_occupancy as so
except:
    from . import spectral_occupancy as so

def reduce_frames(csv_list):
    df = pd.DataFrame()
    for csv in csv_list:
        temp = pd.read_csv(csv)
        df = df.append(temp, ignore_index=True)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generates a histogram of the spectral occupancy from a given set of .dat files")
    parser.add_argument("band", help="the GBT band that the data was collected from. Either L, S, C, or X")
    parser.add_argument("csv_dir", help="directory .csv files are held in")
    # parser.add_argument("h5_dir", help="directory .h5 files are held in")
    parser.add_argument("outdir", help="directory where the output cadences searches will be saved")
    # parser.add_argument("-n_iterations", "-n", help="number of random cadences to generate and search, default is 1 random shuffle", default=1)
    # parser.add_argument("-t", help="a .txt file to read the filepaths of the .dat files", action=None)
    # parser.add_argument("-width", "-w", help="width of bin in Mhz", type=float, default=1)
    # parser.add_argument("-notch_filter", "-nf", help="exclude data that was collected within GBT's notch filter when generating the plot", action="store_true")
    # parser.add_argument("-DC", "-d", help="files contain DC spikes that need to be removed", action="store_true")
    # parser.add_argument("-save", "-s", help="save the histogram bin edges and heights", action="store_true")
    args = parser.parse_args()

    csv_paths = glob.glob(args.csv_dir)
    df = reduce_frames(csv_paths+"/*/*.csv")

    hist, bin_edges = so.calculate_hist(df, args.band)

    plt.figure(figzise=(10,5))
    plt.bar(bin_edges[:-1], hist, width=1)
    plt.xlabel("Frequency [Mhz]")
    plt.ylabel("Fraction with Hits")
    plt.title("Histogram of Chance Candidates")
    plt.savefig("%s_band_random_chance_candidates.pdf"%args.band, bbox_inches="tight", transparent=False)
    plt.savefig("%s_band_random_chance_candidates.png"%args.band, bbox_inches="tight", transparent=False)