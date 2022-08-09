import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse 
import os
import time
from tqdm import trange
from multiprocessing import Pool

def reduce_frames(csv_list):
    df = pd.DataFrame()
    for i in trange(len(csv_list)):
        temp = pd.read_csv(csv_list[i])
        temp = temp[temp["status"] == "on_table_1"] # reduce hits to only the first target hits
        df = df.append(temp, ignore_index=True)
    return df

def check_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def plot_pipeline(plot_directory):
    plot_dir = plot_directory + "plots/"
    check_dir(plot_dir)

    event_csv_string = plot_directory + "events.csv"
    h5_list_string = plot_directory + "h5_list.lst"

    plot_event_pipeline(event_csv_string,
                        h5_list_string, 
                        user_validation=False,
                        filter_spec="3",
                        sortby_tstart=False,
                        plot_dir=plot_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generates a histogram of the spectral occupancy from a given set of .dat files")
    parser.add_argument("band", help="the GBT band that the data was collected from. Either L, S, C, or X")
    parser.add_argument("csv_dir", help="directory .csv files are held in")
    parser.add_argument("outdir", help="directory where the output cadences searches will be saved")
    parser.add_argument("-waterfall", "-w", help="make waterfall plots of the candidate signals instead of histograms", action="store_true")
    parser.add_argument("-parallel", "-p", help="number of processes to accelerate with", default=1)
    args = parser.parse_args()

    csv_paths = glob.glob(args.csv_dir + "/*/*.csv")
    total_observations = glob.glob(args.csv_dir + "*")

    # make the histogram plots
    if not args.waterfall:
        print("making histogram plots at", (args.outdir + "%s_band_random_chance_candidates.pdf"%args.band))
        try:
            import spectral_occupancy as so
        except:
            from . import spectral_occupancy as so

        df = reduce_frames(csv_paths)

        hist, bin_edges = so.calculate_hist(df, args.band)
        data_dict = {"frequency":bin_edges[:-1], "counts":hist}
        data_df = pd.DataFrame(data_dict)
        data_df.to_csv(args.outdir + "%s_band_random_chance_candidates.csv"%args.band)


        plt.figure(figsize=(10,5))
        plt.bar(bin_edges[:-1], hist, width=1)
        plt.xlabel("Frequency [Mhz]")
        plt.ylabel('Number of Candidate "Signals"')
        plt.title("Histogram of Chance Candidates\nN = %s observations and n = %s with candidates"%(len(total_observations), len(csv_paths)))
        plt.savefig(args.outdir + "%s_band_random_chance_candidates.pdf"%args.band, bbox_inches="tight", transparent=False)
        plt.savefig(args.outdir + "%s_band_random_chance_candidates.png"%args.band, bbox_inches="tight", transparent=False)

    else:
        print("making waterfall plots of events")
        from old_turbo_seti.turbo_seti.find_event.plot_event_pipeline import plot_event_pipeline
        for i in range(len(csv_paths)):
            csv_paths[i] = os.path.dirname(csv_paths[i]) + "/"
        # data_inputs = [(os.path.dirname(path) + "/") for path in csv_paths]

        t_start = time.time()
        pool = Pool(int(args.parallel))
        pool.map(plot_pipeline, csv_paths)
        print("runtime = ", (time.time() - t_start))