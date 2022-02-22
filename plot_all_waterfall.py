import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import blimpy as bl
import gc

def z_score(df):
    """
    Computes the z_score for each entry of the 
    DataFrame. Assumes that each row corresponds 
    to a unizue file, and each column corresponds 
    to a numerical value 
    """
    arr = np.array(df)
    mean = np.mean(arr, axis=0)
    sd = np.std(arr, axis=0)
    no_data = np.where(np.sum(arr, axis=0) == 0)
    # correct for zero standard deviations
    sd[no_data] = np.inf
    mask = np.where(sd==0)
    sd[mask] = 1e-9
    z_tbl = (arr - mean)/sd
    
    return z_tbl

def flag_z(df, min_z, region="upper"):
    """
    Takes a DataFrame and minimum Z-score and returns 
    all the files with channels above this Z-score
    as well as the corresponding frequencies
    
    Will output two lists of the same length. The 
    first file will hold the filenames and the second 
    will hold the frequencies in that file that are 
    above the Z-score
    
    Arguments:
    -----------
    df : pandas.core.frame.DataFrame
        DataFrame containing the fine channel fraction data
    min_z : float
        the minimum allowed Z-score, above which files will
        be flagged
        
    Returns:
    ---------
    flagged_files : pandas.core.indexes.base.Index
        An array containing strings of the filenames
        of the flagged files
    flagged_freqs : list
        A list of frequencies that were flagged 
    """
    z_tbl = z_score(df)
    if region=="upper":
        mask = np.where(z_tbl > min_z)
    else:
        mask = np.where(z_tbl < min_z)
    stacked = np.vstack(mask)
    file_indx = np.unique(stacked[0])
    flagged_freqs = []
    for indx in file_indx:
        this_file = stacked[1][np.where(stacked[0] == indx)[0]]
        frequencies = np.array(df.T.iloc[this_file].index)
        flagged_freqs.append(frequencies)
    flagged_files = df.iloc[file_indx].index
    return flagged_files, flagged_freqs

def sort_flags(df, min_z=2):
    """
    Flags frequency bins above a given z score
    and sorts the files according to how many
    bins are flagged

    Arguments
    ----------
    df : pandas.core.frame.DataFrame
        A DataFrame containing histograms of the detected 
        hits for a large number of observations
    min_z : float
        the minimum allowed z score, above which
        all bins will be flagged a unusual
    
    Returns
    --------
    sorted_df : pandas.core.frame.DataFrame
        A DataFrame containing the filname and number of flagged bins, 
        sorted from least flags to most
    """
    indices = list(df.index)
    flag_count = np.sum((z_score(df) >= min_z), axis=1)

    data_dict = {"filename":indices, "flagged bins":flag_count}
    sorted_df = pd.DataFrame(data_dict)
    return sorted_df.sort_values("flagged bins")

def spectrum_and_waterfall(h5_path, f_start, f_stop, spectrum_path, waterfall_path, max_load=1, show=True):
    """
    Plots and saves the spectrum and waterfall of a given h5 file

    Arguments
    ----------
    h5_path : str
        Path to the source h5 file, which will be used to
        generate the plots
    f_start : float
        Smallest frequency that will be plotted. Units are MHz
    f_stop : float
        Highest frequency that will be plotted. Units are MHz
    spectrum_path : str
        Filepath to where the generated spectrum plot will be saved
    waterfall_path : str
        Filepath to where the generated waterfall plot will be saved
    max_load : float
        Size of the file that will be loaded
    show : bool
        Option to show the plots generated. Setting to false allows for 
        faster runtime and less load on memory

    Returns None
    """
    hf = bl.Waterfall(h5_path, f_start=f_start, f_stop=f_stop, max_load=max_load)
    print("data loaded")
    plt.figure(figsize=(20,10))
    hf.plot_spectrum()
    plt.tight_layout()
    plt.savefig(spectrum_path, bbox_inches='tight', transparent=False)
    if not show:
        plt.close("all")

    plt.figure(figsize=(20,10))
    hf.plot_waterfall()
    plt.tight_layout()
    plt.savefig(waterfall_path, bbox_inches='tight', transparent=False)
    del hf
    gc.collect()
    if not show:
        plt.close("all")
    return 

if __name__ == "__main__":
    # paths to observation histograms
    lband_csv = "/home/danielb/fall_2021/histograms/energy_detection_csvs/L_band_ALL_energy_detection_hist_threshold_4096_with_notch_data.csv'"
    sband_csv = "/home/danielb/fall_2021/histograms/energy_detection_csvs/S_band_ALL_energy_detection_hist_threshold_4096_with_notch_data.csv"
    cband_csv = "/home/danielb/fall_2021/histograms/energy_detection_csvs/C_band_ALL_energy_detection_hist_threshold_4096.csv"
    xband_csv = "/home/danielb/fall_2021/histograms/energy_detection_csvs/X_band_ALL_energy_detection_hist_threshold_4096.csv"
    band_csvs = [lband_csv, sband_csv, cband_csv, xband_csv]

    