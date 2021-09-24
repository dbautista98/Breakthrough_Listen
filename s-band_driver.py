# driver function for calling energy detection on S-band data

import os
import glob

def run_energy_detection(filepath):
    energy_detection = "python3 BL-Reservoir/energy_detection/energy_detection_fine_dry_run_CSV.py "
    file_name = os.path.basename(filepath)
    folder_name = " energy_detection/"+file_name.replace(".gpuspec.0000.h5", "")
    os.system(energy_detection + filepath + folder_name)
    print("completed another file\n\n")
    with open("completed_s-band.txt", "a") as f:
        f.write(file+"\n")
    
all_files = glob.glob("/home/dbautista98/data/s_band/*0.h5")

for file in all_files:
    run_energy_detection(file)
    
    
print("ALL DONE")
