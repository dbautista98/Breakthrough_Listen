import os
import numpy as np



bands = ["L", "S"]
thresholds = 2**np.arange(13)

for band in bands:
    for threshold in thresholds:
        lowercase = band.lower()
        output = "python3 /home/dbautista98/Breakthrough_Listen/GCP/compute_hist.py %s /home/dbautista98/energy-detection/%s-band/ -t %s -o histograms/%s-band/ -nf"%(band, lowercase, threshold, lowercase)
        print(output)
        os.system(output)

print("ALL DONE")
