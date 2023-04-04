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
    parser = argparse.ArgumentParser(description="randomly selects and runs turboSETI's event search on a set of .dat files. This is done to estimate the chances of geting a technosignature event from random RFI")
    parser.add_argument("dat_dir", help="directory .dat files are held in")
    parser.add_argument("h5_dir", help="directory .h5 files are held in")
    parser.add_argument("outdir", help="directory where the output cadences searches will be saved")
    parser.add_argument("-n_iterations", "-n", help="number of random cadences to generate and search, default is 1 random shuffle", default=1)
    parser.add_argument("-parallel", "-p", help="number of parallel processes to accelerate with", default=1)
    args = parser.parse_args()

    # random_cadence_search(args.dat_dir, args.h5_dir, args.outdir, int(args.n_iterations))
    pool = Pool(int(args.parallel))                         # Create a multiprocessing Pool
    t_start = time.time()
    data_inputs = [[args.dat_dir,args.h5_dir,args.outdir,i] for i in range(int(args.n_iterations))]
    pool.map(random_cadence_search_parallel, data_inputs)
    print("runtime =", (time.time() - t_start))


                    