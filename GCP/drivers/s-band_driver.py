# driver function for calling energy detection on S-band data

import os
import glob
import numpy as np

def run_energy_detection(filepath):
    energy_detection = "python3 BL-Reservoir/energy_detection/energy_detection_fine_dry_run_CSV.py "
    file_name = os.path.basename(filepath)
    folder_name = " energy-detection/s-band/"+file_name.replace(".gpuspec.0000.h5", "")
    os.system(energy_detection + filepath + folder_name)
    print(file_name)
    with open("completed_s-band.txt", "a") as f:
        f.write(file_name+"\n")
    
all_files = glob.glob("/home/dbautista98/data/s_band/*0.h5")
total_files = len(all_files)

for i, file in enumerate(all_files):
    run_energy_detection(file)
    percent = (i+1)/total_files*100
    print(np.round(percent,1), "% of files completed\n\n")
    
    
print("ALL DONE")
