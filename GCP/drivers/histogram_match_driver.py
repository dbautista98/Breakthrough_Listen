import os
import numpy as np



bands = ["L", "S"]

for band in bands:
    lowercase = band.lower()
    output = "python3 /home/dbautista98/Breakthrough_Listen/GCP/histogram_match.py %s /home/dbautista98/energy-detection/%s-band/ /home/dbautista98/histograms/turboSETI/%s_band_turboSETI_hist.csv -o /home/dbautista98/histograms/threshold_comparisons -nf -p"%(band, lowercase, band)
    print(output)
    os.system(output)

print("ALL DONE")
