import os
import numpy as np



bands = ["X", "L", "S", "C"]
thresholds = 2**np.arange(16)

i = 0
while True:
    band = bands[i%4]
    for threshold in thresholds:
        lowercase = band.lower()
        output = "python3 /home/dbautista98/Breakthrough_Listen/GCP/compute_hist.py %s /home/dbautista98/energy-detection/%s-band/ -t %s -o histograms/%s-band/ -nf"%(band, lowercase, threshold, lowercase)
        print(output)
        os.system(output)
        i += 1

print("ALL DONE")
