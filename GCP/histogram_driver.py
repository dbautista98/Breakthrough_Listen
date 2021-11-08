import os
import numpy as np



bands = ["L", "S"]
thresholds = 2**np.arange(13)

for band in bands:
    for threshold in thresholds:
        lowercase = band.lower()
        output = "python3 compute_hist.py %s /home/dbautista98/energy-detection/%s-band/ -t %s -o histograms/%s-band/"%(band, lowercase, threshold, lowercase)
        print(output)
        os.system(output)

print("ALL DONE")
