import os 
import numpy as np


bands = ["X", "C"]
thresholds = 2**np.arange(16)
energy_dir = "/home/dbautista98/energy-detection/" 
script_path = "/home/dbautista98/Breakthrough_Listen/GCP/spectral_occupancy.py"


for band in bands:
    for thresh in thresholds:
        data_path = energy_dir+"%s-band/"%band.lower()
        occupancy_call = "python3 " + script_path + (" %s "%band) + " " + data_path + " -nf -s -t %s"%thresh
        print(occupancy_call)
        os.system(occupancy_call)
