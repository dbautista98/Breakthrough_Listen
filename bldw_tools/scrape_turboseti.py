"""
This program walks through /home/obs/turboseti/
and writes all existing .dat files to a .txt file
in /datax/scratch/danielb/GBT_statistics/all_bldw/

These file paths can then be split up into bands and
used in spectral occupancy calculations
"""

import os

turboseti = "/home/obs/turboseti/"
dump = "/datax/scratch/danielb/GBT_statistics/all_bldw/dump.txt"

with open(dump, "w") as f:
    pass

shpfiles = []
for dirpath, subdirs, files in os.walk(turboseti):
    for x in files:
        if x.endswith(".dat"):
            path = os.path.join(dirpath, x)
            if os.path.exists(path):
                with open(dump, "a") as f:
                    f.write(path + "\n")