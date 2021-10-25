import os
import numpy as np



band = "L"
lowercase = band.lower()
thresholds = 2**np.arange(13)

for threshold in thresholds:
    output = "python3 compute_hist.py %s /home/dbautista98/energy-detection/%s-band/ -t %s -o histograms/%s-band/"%(band, lowercase, threshold, lowercase)
    print(output)
    os.system(output)

print("ALL DONE")
