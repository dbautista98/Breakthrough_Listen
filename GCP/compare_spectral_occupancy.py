import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse

def read_pickle(pickle_path):
    """
    takes in path to pickle file and returns its contents

    Arguments:
    -----------
    pickle_path : str
        filepath to the pickled file

    Returns:
    ---------
    contents : type = ???
        the saved contents of the pickle file, data type is 
        dependant on what was put into the pickle file
    """

    with open(pickle_path, "rb") as f:
        contents = pickle.load(f)
    return contents

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compares the contents of two pickle files containing the spectral occupancy data from different processing algorithms")
    parser.add_argument("turboSETI", help="filepath to the .pkl file with turboSETI spectral occupancy")
    parser.add_argument("energy_detection", help="filepath to the .pkl file with energy detection spectral occupancy")
    args = parser.parse_args()

    turboSETI = read_pickle(args.turboSETI)
    energy = read_pickle(args.energy_detection)

    checks = np.ones(4)
    # sanity checks that the data are comparable
    if turboSETI["algorithm"] != "turboSETI":
        print("You did not pass in a turboSETI file")
    else:
        checks[0] = 0

    if energy["algorithm"] != "energy detection":
        print("You did not pass in an energy detection file")
    else:
        checks[1] = 0

    if turboSETI["band"] != energy["band"]:
        print("These are not from the same band")
    else:
        checks[2] = 0

    if turboSETI["bin width"] != energy["bin width"]:
        print("The bin sizes are not the same")
    else:
        checks[3] = 0

    if np.sum(checks) == 0:
        turbo_heights = turboSETI["bin_heights"]
        energy_heights= energy["bin_heights"]
        midline = np.linspace(0,1)

        plt.figure(figsize=(10,7))
        plt.plot(midline, midline, label="y=x line", color="black")
        plt.scatter(turbo_heights, energy_heights)
        plt.xlabel("turboSETI bin height")
        plt.ylabel("energy detection bin height")
        plt.legend()
        plt.title("%s band comparison of %s energy detection files at a threshold of %s\nand %s turboSETI files, with a bin width of %s MHz"%(energy["band"], energy["n files"], energy["threshold"], turboSETI["n files"], turboSETI["bin width"]))
        plt.savefig("%s_band_%s_MHz_bins_%s_threshold.pdf"%(energy["band"], energy["bin width"], energy["threshold"]))
    
