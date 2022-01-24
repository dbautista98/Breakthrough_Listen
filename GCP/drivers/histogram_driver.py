import os
import numpy as np



bands = ["L", "S", "X", "C"]
thresholds = [4096]#2**np.arange(16)

for band in bands:
    for threshold in thresholds:
        lowercase = band.lower()
        output = "python3 /home/dbautista98/Breakthrough_Listen/GCP/compute_hist.py %s /home/dbautista98/energy-detection/%s-band/ -t %s -o /home/dbautista98/histograms/energy_detection_csvs/ -nf -s"%(band, lowercase, threshold)
        print(output)
        os.system(output)

print("ALL DONE")
