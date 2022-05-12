import glob
import pandas as pd
import os 

def get_data():
    current_dir = __file__.replace(os.path.basename(__file__), "")
    csvs = glob.glob(current_dir + "*csv")
    csvs.sort()

    L = pd.read_csv(csvs[1], index_col="filename")
    S = pd.read_csv(csvs[2])
    C = pd.read_csv(csvs[0])
    X = pd.read_csv(csvs[3])

    return {"L":L, "S":S, "C":C, "X":X}