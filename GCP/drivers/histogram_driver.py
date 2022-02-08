import os
import numpy as np


"""
DESCRIPTION:

This program will compute the histograms of all the energy detection outputs and checks them against files 
with missing compute nodes. It also saves a csv of the corresponding histograms. Additionally, it will save
a csv containing the histogram of each file in the dataset
"""


bands = ["L", "S", "X", "C"]
thresholds = [4096]#2**np.arange(16)
dropped_csv = "/home/dbautista98/dropped_nodes.csv"

for band in bands:
    for threshold in thresholds:
        lowercase = band.lower()
        output = "python3 /home/dbautista98/Breakthrough_Listen/GCP/compute_hist.py %s /home/dbautista98/energy-detection/%s-band/ -t %s -o /home/dbautista98/histograms/energy_detection_csvs/ -nf -s -c %s"%(band, lowercase, threshold, dropped_csv)
        print(output)
        os.system(output)

print("ALL DONE")
