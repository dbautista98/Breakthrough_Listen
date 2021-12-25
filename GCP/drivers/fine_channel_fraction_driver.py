import os 
import numpy as np


bands = ["X", "C", "S"]
energy_dir = "/home/dbautista98/energy-detection/" 
script_path = "/home/dbautista98/Breakthrough_Listen/GCP/fine_channel_fraction.py"

for band in bands:
    data_path = energy_dir+"%s-band/"%band.lower()
    program_call = "python3 " + script_path + (" %s "%band) + data_path 
    print(program_call)
    os.system(program_call)
