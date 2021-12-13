import os 
import numpy as np

bands = ["X", "C"]
thresholds = 2**np.arange(16)
data_dir = "/home/dbautista98/spectral_occupancy/" 
script_path = "/home/dbautista98/Breakthrough_Listen/GCP/compare_spectral_occupancy.py"

for band in bands: 
    for thresh in thresholds:
        turboSETI_path = data_dir + "turbo-seti/turboSETI_%s_band_spectral_occupancy_1_MHz_bins.pkl"%band
        energy_det_path= data_dir + "energy-detection/energy_detection_%s_band_spectral_occupancy_1_MHz_bins_%s.0_threshold.pkl"%(band, thresh)
        sys_call = "python3 " + script_path + " " + turboSETI_path + " " + energy_det_path
        os.system(sys_call)
