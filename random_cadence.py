import glob
import numpy as np
import pickle
import os
from old_turbo_seti.turbo_seti.find_event.find_event_pipeline import find_event_pipeline
from old_turbo_seti.turbo_seti.find_event.plot_event_pipeline import plot_event_pipeline
import argparse
from multiprocessing import Pool

def open_pickle(filepath):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data

def check_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def random_cadence_search(dat_dir, h5_dir, outdir, n_iterations=1):
    np.random.seed(seed=42)
    for i in range(n_iterations):
        dat_list = glob.glob(dat_dir + "/*dat")
        shuffled_cadence = np.random.choice(dat_list, size=6, replace=False)
        check_dir(outdir)
        check_dir(outdir +"/%s/"%i)
        iteration_directory = outdir + "/%s/"%i

        with open(iteration_directory + "dat_list.lst", "w") as f:
            for file in shuffled_cadence:
                f.write(file + "\n")

        with open(iteration_directory + "h5_list.lst", "w") as f:
            for file in shuffled_cadence:
                base_name_to_h5 = (os.path.basename(file)).replace("dat", "h5")
                h5_path = h5_dir + base_name_to_h5
                f.write(h5_path + "\n")

        find_event_pipeline(iteration_directory + "dat_list.lst",
                        filter_threshold = 3,
                        number_in_cadence = len(shuffled_cadence),
                        user_validation=False,
                        saving=True,
                        csv_name=iteration_directory + "events.csv",
                        sortby_tstart=False,
                        h5_dir=h5_dir)

def random_cadence_search_parallel(data_input):
    dat_dir, h5_dir, outdir, i = data_input
    np.random.seed(seed=i)
    dat_list = glob.glob(dat_dir + "/*dat")
    shuffled_cadence = np.random.choice(dat_list, size=6, replace=False)
    check_dir(outdir)
    check_dir(outdir +"/%s/"%i)
    iteration_directory = outdir + "/%s/"%i

    with open(iteration_directory + "dat_list.lst", "w") as f:
        for file in shuffled_cadence:
            f.write(file + "\n")

    with open(iteration_directory + "h5_list.lst", "w") as f:
        for file in shuffled_cadence:
            base_name_to_h5 = (os.path.basename(file)).replace("dat", "h5")
            h5_path = h5_dir + base_name_to_h5
            f.write(h5_path + "\n")

    find_event_pipeline(iteration_directory + "dat_list.lst",
                    filter_threshold = 3,
                    number_in_cadence = len(shuffled_cadence),
                    user_validation=False,
                    saving=True,
                    csv_name=iteration_directory + "events.csv",
                    sortby_tstart=False,
                    h5_dir=h5_dir)

if __name__ == "__main__":
    import time
    parser = argparse.ArgumentParser(description="generates a histogram of the spectral occupancy from a given set of .dat files")
    # parser.add_argument("band", help="the GBT band that the data was collected from. Either L, S, C, or X")
    parser.add_argument("dat_dir", help="directory .dat files are held in")
    parser.add_argument("h5_dir", help="directory .h5 files are held in")
    parser.add_argument("outdir", help="directory where the output cadences searches will be saved")
    parser.add_argument("-n_iterations", "-n", help="number of random cadences to generate and search, default is 1 random shuffle", default=1)
    parser.add_argument("-parallel", "-p", help="number of parallel processes to accelerate with", default=1)
    # parser.add_argument("-t", help="a .txt file to read the filepaths of the .dat files", action=None)
    # parser.add_argument("-width", "-w", help="width of bin in Mhz", type=float, default=1)
    # parser.add_argument("-notch_filter", "-nf", help="exclude data that was collected within GBT's notch filter when generating the plot", action="store_true")
    # parser.add_argument("-DC", "-d", help="files contain DC spikes that need to be removed", action="store_true")
    # parser.add_argument("-save", "-s", help="save the histogram bin edges and heights", action="store_true")
    args = parser.parse_args()

    # random_cadence_search(args.dat_dir, args.h5_dir, args.outdir, int(args.n_iterations))
    pool = Pool(int(args.parallel))                         # Create a multiprocessing Pool
    t_start = time.time()
    data_inputs = [[args.dat_dir,args.h5_dir,args.outdir,i] for i in range(int(args.n_iterations))]
    pool.map(random_cadence_search_parallel, data_inputs)
    print("runtime =", (time.time() - t_start))


                    