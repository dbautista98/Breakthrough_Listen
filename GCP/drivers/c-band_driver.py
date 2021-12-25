# driver function for calling energy detection on C-band data

import os
import glob
import numpy as np
import os

band = "c"

def run_energy_detection(filepath):
    try:
        energy_detection = "python3 BL-Reservoir/energy_detection/generalized_energy_detection_fine_dry_run_CSV.py "
        file_name = os.path.basename(filepath)
        folder_name = (" energy-detection/%s-band/"%band)+file_name.replace(".gpuspec.0000.h5", "").replace(".rawspec.0000.h5", "")
        os.system(energy_detection + filepath + folder_name)
        print("Completed %s"%file_name)
        return True
    except:
        print("Failed %s"%file_name)
        return False

def path_to_list(path):
    with open(path, 'r') as f:
        contents = f.read()
    lst = contents.split("\n")
    return lst

def get_unprocessed_files(data_path, completed_path):
    if os.path.exists(completed_path):
        completed_files = path_to_list(completed_path)
        all_files = glob.glob(data_path)

        remaining_files = []
        for a_file in all_files:
            if a_file not in completed_files:
                remaining_files.append(a_file)
        return remaining_files
    else:
        return glob.glob(data_path)

data_path = "/home/dbautista98/data/%s_band/*0.h5"%band
all_files = glob.glob(data_path)
total_files = len(all_files)
completed_path = "/home/dbautista98/completed_%s-band.txt"%band
failed_path = "/home/dbautista98/failed_%s-band.txt"%band
files_to_process = get_unprocessed_files(data_path, completed_path)

for i, a_file in enumerate(files_to_process):
    success = run_energy_detection(a_file)
    if success:
        with open(completed_path, "a") as f:
            f.write(a_file+"\n")
    else:
        with open(failed_path, "a") as f:
            f.write(a_file+"\n")
    percent = (len(path_to_list(completed_path)) - 1)/total_files*100
    print(np.round(percent,2), "% of files completed\n\n")
    
    
print("ALL DONE")
