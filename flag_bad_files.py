import pandas as pd
import pandas
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import glob
import os
import argparse
try:
    import spectral_occupancy as so
except:
    from . import spectral_occupancy as so
try:
    import remove_DC_spike as dc
except:
    from . import remove_DC_spike as dc
import turbo_seti.find_event as fe
from scipy.optimize import curve_fit

def node_boundaries(band, output="default"):
    """
    returns the compute nodes and their boundaries for an observing band

    Arguments
    ----------
    band : str
        the band at which the data was collected
        choose from {"L", "S", "C", "X"}
    output : str
        option for the output form of the data.
        `default` returns nodes, boundaries 
        `dict` returns a dictionary containing the pairs
    
    Returns
    --------
    nodes : list
        the name of the compute node which the data
        was processed on
    boundaries : numpy.ndarray
        the filterband fch1, as reported in Lebofsky et al 2019
    """
    if band.upper() == "L":
        boundaries = np.asarray([2251.46484375, 2063.96484375, 1876.46484375, 1688.96484375, 
                                 1501.46484375, 1313.96484375, 1126.46484375, 938.96484375])

    if band.upper() == "S":
        boundaries = np.asarray([3151.46484375, 2963.96484375, 2776.46484375, 2588.96484375, 
                                 2401.46484375, 2213.96484375, 2026.46484375, 1838.96484375])

    if band.upper() == "C":
        boundaries = np.asarray([8438.96484375, 8251.46484375, 8063.96484375, 7876.46484375,
                                 7688.96484375, 7501.46484375, 7313.96484375, 7126.46484375, 
                                 7313.96484375, 7126.46484375, 6938.96484375, 6751.46484375, 
                                 6563.96484375, 6376.46484375, 6188.96484375, 6001.46484375, 
                                 6188.96484375, 6001.46484375, 5813.96484375, 5626.46484375, 
                                 5438.96484375, 5251.46484375, 5063.96484375, 4876.46484375, 
                                 5063.96484375, 4876.46484375, 4688.96484375, 4501.46484375, 
                                 4313.96484375, 4126.46484375, 3938.96484375, 3751.46484375])
    
    if band.upper() == "X":
        boundaries = np.asarray([11251.4648437, 11063.9648437, 10876.4648437, 10688.9648437,
                                 10501.4648437, 10313.9648437, 10126.4648437, 9938.96484375, 
                                 10126.4648437, 9938.96484375, 9751.46484375, 9563.96484375, 
                                 9376.46484375, 9188.96484375, 9001.46484375, 8813.96484375, 
                                 9001.46484375, 8813.96484375, 8626.46484375, 8438.96484375, 
                                 8251.46484375, 8063.96484375, 7876.46484375, 7688.96484375])

    all_nodes = ['blc00', 'blc01', 'blc02', 'blc03', 'blc04', 'blc05', 'blc06', 'blc07', 
                 'blc10', 'blc11', 'blc12', 'blc13', 'blc14', 'blc15', 'blc16', 'blc17', 
                 'blc20', 'blc21', 'blc22', 'blc23', 'blc24', 'blc25', 'blc26', 'blc27', 
                 'blc30', 'blc31', 'blc32', 'blc33', 'blc34', 'blc35', 'blc36', 'blc37']
    
    if output == "dict":
        out_dict = {}
        for i in range(len(boundaries)):
            out_dict[all_nodes[:len(boundaries)][i]] = boundaries[i]
        return out_dict

    return all_nodes[:len(boundaries)], boundaries

def select_node(df, fch1):
    """
    returns only the data within one compute node

    Arguments
    ----------
    df : pandas.core.frame.DataFrame
        DataFrame containing processed data from an observation
    fch1 : float
        the first frequency recorded in a compute node
    
    Returns
    --------
    reduced_df : pandas.core.frame.DataFrame
        a DataFrame containing only the data within a given compute node
    """
    mask = np.where((df["frequency"] <= fch1) & (df["frequency"] > (fch1 - 187.5)))
    reduced_df = df.iloc[mask]
    return reduced_df

def check_possible_data(fch1, band_edges):
    """
    checks if the value of fch1 is within the 
    frequency bounds of what is considered useful data

    Arguments
    ----------
    fch1 : float
        center frequency of the first fine channel
    band_edges : numpy.ndarray
        array containing the highest and lowest frequencies
        that will be considered for the band
    
    Returns
    --------
    in_range : bool
        status of whether or not fch1 is within the given frequency bounds
    """

    in_range = (fch1 < np.max(band_edges)) and ((fch1) > np.min(band_edges))
    return in_range

def energy_detection_boxcar_analysis(df, nodes, boundaries):
    """
    steps through the frequency intervals corresponding to each 
    compute node and calculates the mean and standard deviation 
    of the data in the interval

    Arguments
    ----------
    df : pandas.core.frame.DataFrame
        DataFrame containing processed data from an observation
    nodes : list
        the name of the compute node which the data
        was processed on
    boundaries : numpy.ndarray
        the filterband fch1, as reported in Lebofsky et al 2019

    Returns
    --------
    result_df : pandas.core.frame.DataFrame
        DataFrame summarizing the data within across 
        all compute nodes
    """
    means = np.empty_like(boundaries)
    st_devs=np.empty_like(boundaries)
    data_present = np.empty_like(boundaries, dtype=bool)

    for i in range(len(boundaries)):
        fch1 = boundaries[i]
        df_subset = select_node(df, fch1)
        bins = np.linspace(fch1-187.5, fch1, num=1875, endpoint=True) # bin size of 0.1 MHz
        hist, bin_edges = np.histogram(df_subset["frequency"].values, bins=bins)
        if len(df_subset) == 0:
            means[i] = 0
            st_devs[i] = 0
            data_present[i] = False
        elif len(df_subset) == 1:
            means[i] = df_subset["statistic"].values[0]
            st_devs[i] = 1e9
            data_present[i] = True
        elif len(df_subset) == 2:
            means[i] = np.mean(df_subset["statistic"].values)
            st_devs[i] = 1e9
            data_present[i] = True
        else:
            means[i] = np.mean(df_subset["statistic"].values)#np.mean(hist)
            st_devs[i] = np.std(df_subset["statistic"].values)#np.std(hist)
            data_present[i] = True
        
        data_dict = {"nodes":nodes, "fch1":boundaries, "data present":data_present, "mean statistic":means, "standard deviation":st_devs}
        result_df = pd.DataFrame(data_dict)

    return result_df

def turbo_seti_boxcar_analysis(df, nodes, boundaries, band):
    """
    steps through the frequency intervals corresponding to each 
    compute node and calculates the mean and standard deviation 
    of the data in the interval

    Arguments
    ----------
    df : pandas.core.frame.DataFrame
        DataFrame containing processed data from an observation
    nodes : list
        the name of the compute node which the data
        was processed on
    boundaries : numpy.ndarray
        the filterband fch1, as reported in Lebofsky et al 2019
    band : str
        the band at which the data was collected
        choose from {"L", "S", "C", "X"}

    Returns
    --------
    result_df : pandas.core.frame.DataFrame
        DataFrame summarizing the data within across 
        all compute nodes
    """
    data_present = np.empty_like(boundaries, dtype=bool)
    band_edges = np.array(so.band_edges(band))
    
    for i in range(len(boundaries)):
        fch1 = boundaries[i]
        df_subset = select_node(df, fch1)
        possible_data = check_possible_data(fch1, band_edges)
        if not possible_data:
            data_present[i] = True
        else:
            data_present[i] = (len(df_subset) != 0)
        
    data_dict = {"nodes":nodes, "fch1":boundaries, "data present":data_present}
    result_df = pd.DataFrame(data_dict)
    return result_df

def format_energy_detection(df, threshold=4096):
    """
    renames energy detection columns and makes threshold cuts

    Arguments
    ----------
    df : pandas.core.frame.DataFrame
        DataFrame containing processed data from an observation
    threshold : float
        The minimum value of an allowed hit, below which
        all will be rejected

    Returns
    --------
    reduced_df : pandas.core.frame.DataFrame
        DataFrame with renamed frequency column and 
        thresholded above the given statistic value
    """
    df.rename(columns={"freqs":"frequency"}, inplace=True)
    mask = np.where(df["statistic"] >= threshold)
    reduced_df = df.iloc[mask]
    return reduced_df

def format_turbo_seti(df):
    """
    renames the frequency column to be compatible with previous outputs

    Arguments
    ----------
    df : pandas.core.frame.DataFrame
        DataFrame containing processed data from an observation

    Returns
    --------
    df : pandas.core.frame.DataFrame
        DataFrame with renames frequency column
    """
    df.rename(columns={"Freq":"frequency"}, inplace=True)
    return df

def energy_detection_check_missing(results_df):
    """
    uses the file summary data to determine if 
    there were any dropped compute nodes 

    Arguments
    ----------
    results_df : pandas.core.frame.DataFrame
        DataFrame summarizing the data within across 
        all compute nodes
    
    Returns
    --------
    dropped_nodes : str
        a bitmap string of which nodes are dropped
        starting with blc00 and ending on the highest blcXX
    """
    nodes = results_df["nodes"].values
    has_data = results_df["data present"].values
    sd = results_df["standard deviation"].values
    node_drops = []
    for i in range(len(nodes)):
        if has_data[i] and np.isclose(sd[i], 0):
            node_drops.append(0)
        else:
            node_drops.append(1)
    
    # convert bitmap to string
    string_nodes = [str(int) for int in node_drops]
    bitmap_string = "".join(string_nodes)
    return bitmap_string

def turbo_seti_check_missing(results_df):
    """
    uses the file summary data to determine if 
    there were any dropped compute nodes 

    Arguments
    ----------
    results_df : pandas.core.frame.DataFrame
        DataFrame summarizing the data within across 
        all compute nodes
    
    Returns
    --------
    dropped_nodes : str
        a bitmap string of which nodes are dropped
        starting with blc00 and ending on the highest blcXX
    """
    nodes = results_df["nodes"].values
    has_data = results_df["data present"].values
    node_drops = []
    
    for i in range(len(nodes)):
        if has_data[i]:
            node_drops.append(1)
        else:
            node_drops.append(0)
            
    # convert bitmap to string
    string_nodes = [str(int) for int in node_drops]
    bitmap_string = "".join(string_nodes)
    return bitmap_string

def identify_missing_node(bitmap, nodes):
    """
    Use the bitmap to identify which nodes are missing

    Arguments
    ----------
    bitmap : str
        A string of ones and zeros (eg: '000100001') indicating whether 
        a node has been dropped during analysis. A (0) indicates that the 
        node was dropped and a (1) indicates that the node was included and
        that nothing went wrong 
    nodes : list
        the name of the compute node which the data was processed on
    
    Returns
    --------
    node_list : list
        a list of all the nodes that were dropped during data recording
    """
    bit_list = np.array(list(bitmap))
    mask = (bit_list == "0")
    return list(np.array(nodes)[mask])

def get_target_name(filename):
    """
    picks the name of the source target based
    on the naming convention of the h5 files

    Arguments
    ----------
    filename : str
        string containing the complete filepath to the data

    Returns
    --------
    source_name : str
        the name of the source observed in the given file
    """
    base_name = filename.split("/")
    file_name = base_name[-1]
    target_name = file_name.split("_")
    return target_name[-2]

def energy_detection_file_summary(csv_path, band, source_file_name, threshold=4096):
    """
    Determines if an energy detection file is missing any 
    nodes. If the file is missing a node(s) this function 
    will return a pandas DataFrame containing the filename, 
    band, and a bitmap showing which node(s) dropped

    Arguments
    ----------
    csv_path : str
        filepath to the energy detection csv 
    band : str
        the band at which the data was collected
        choose from {"L", "S", "C", "X"}
    source_file_name : str
        the name of the original .h5 file that the data comes from
    threshold : float
        The minimum value of an allowed hit, below which
        all will be rejected

    Returns
    --------
    summary_df : pandas.core.frame.DataFrame
        A pandas DataFrame containing the information 
        about the missing nodes. There are two possible
        outputs: 
           1)  For a file with no dropped nodes, this 
               function will return an empty DataFrame

           2)  A DataFrame containing information about
               the missing nodes, formatted as follows

               filename         band       dropped node bitmap        algorithm
                 filename.h5     L            00010001                  energy detection
    """
    nodes, boundaries = node_boundaries(band)
    df = pd.read_csv(csv_path)
    df = format_energy_detection(df, threshold=threshold)
    results = energy_detection_boxcar_analysis(df, nodes, boundaries)
    missing_string = energy_detection_check_missing(results)
    dropped_node_list = identify_missing_node(missing_string, nodes)
    if len(dropped_node_list) == 0:
        # return an empty DataFrame
        return pd.DataFrame()
    # target_names = []
    # for target in source_file_name:
    #     target_names.append(get_target_name(target))
    summary_dict = {"filename":source_file_name, "band":band.upper(), "dropped node bitmap":missing_string, "dropped node":(" ".join(dropped_node_list)), "algorithm":"energy detection"}
    temp_df = pd.DataFrame()
    summary_df = temp_df.append(summary_dict, ignore_index=True)
    return summary_df[["filename", "band", "dropped node bitmap", "dropped node", "algorithm"]]

def turbo_seti_file_summary(dat_path, band, source_file_name):
    """
    Determines if an energy detection file is missing any 
    nodes. If the file is missing a node(s) this function 
    will return a pandas DataFrame containing the filename, 
    band, and a bitmap showing which node(s) dropped

    Arguments
    ----------
    dat_path : str
        filepath to the dat file
    band : str
        the band at which the data was collected
        choose from {"L", "S", "C", "X"}
    source_file_name : str
        the name of the original .h5 file that the data comes from

    Returns
    --------
    summary_df : pandas.core.frame.DataFrame
        A pandas DataFrame containing the information 
        about the missing nodes. There are two possible
        outputs: 
           1)  For a file with no dropped nodes, this 
               function will return an empty DataFrame

           2)  A DataFrame containing information about
               the missing nodes, formatted as follows

               filename         band       dropped node bitmap        algorithm
                 filename.h5     L            00010001                  turboSETI
    """
    nodes, boundaries = node_boundaries(band)
    df = fe.read_dat(dat_path)
    df = format_turbo_seti(df)
    results = turbo_seti_boxcar_analysis(df, nodes, boundaries, band)
    missing_string = turbo_seti_check_missing(results)
    dropped_node_list = identify_missing_node(missing_string, nodes)
    if len(dropped_node_list) == 0:
        # return an empty DataFrame
        return pd.DataFrame()
    # target_names = []
    # for target in source_file_name:
    #     target_names.append(get_target_name(target))
    summary_dict = {"filename":source_file_name, "band":band.upper(), "dropped node bitmap":missing_string, "dropped node":(" ".join(dropped_node_list)), "algorithm":"turbo_seti"}
    temp_df = pd.DataFrame()
    summary_df = temp_df.append(summary_dict, ignore_index=True)
    return summary_df[["filename", "band", "dropped node bitmap", "dropped node", "algorithm"]]

def check_many_energy_detection_files(missing_files_df, data_list, source_list, band_list, threshold=4096):
    """
    loops over many files and each checks for missing nodes

    Arguments
    ----------
    missing_files_df : pandas.core.frame.DataFrame
        DataFrame of known files missing nodes. If no files
        are known, pass in an empty DataFrame
    data_list : list
        list of paths to the csv or dat files to be checked
    source_list : list
        list file names of the corresponding h5 files
    band_list : list
        list of the corresponding bands the data was collected from
    threshold : float
        The minimum value of an allowed hit, below which
        all will be rejected

    Returns
    --------
    missing_files_df : pandas.core.frame.DataFrame
        DataFrame of all the files flagged for missing nodes
    """
    for i in trange(len(data_list)):
        one_file_df = energy_detection_file_summary(data_list[i], band_list[i], source_list[i], threshold=threshold)
        missing_files_df = missing_files_df.append(one_file_df, ignore_index=True)
    return missing_files_df

def check_many_turbo_seti_files(missing_files_df, data_list, source_list, band_list):
    """
    loops over many files and each checks for missing nodes

    Arguments
    ----------
    missing_files_df : pandas.core.frame.DataFrame
        DataFrame of known files missing nodes. If no files
        are known, pass in an empty DataFrame
    data_list : list
        list of paths to the csv or dat files to be checked
    source_list : list
        list file names of the corresponding h5 files
    band_list : list
        list of the corresponding bands the data was collected from

    Returns
    --------
    missing_files_df : pandas.core.frame.DataFrame
        DataFrame of all the files flagged for missing nodes
    """
    for i in trange(len(data_list)):
        one_file_df = turbo_seti_file_summary(data_list[i], band_list[i].upper(), source_list[i])
        missing_files_df = missing_files_df.append(one_file_df, ignore_index=True)
    return missing_files_df

def energy_detection_driver(missing_files_df, data_dir, threshold=4096):
    """
    function that checks many files and makes a list
    of which files suffer from data loss due to dropped nodes

    Arguments
    ----------
    missing_files_df : pandas.core.frame.DataFrame
        DataFrame of known files missing nodes. If no files
        are known, pass in an empty DataFrame
    data_dir : str
        path to the directory containing the folders
        with the results of running the energy detection 
        algorithm on many files
    threshold : float
        The minimum value of an allowed hit, below which
        all will be rejected

    Returns
    --------
    missing_files_df : pandas.core.frame.DataFrame
        DataFrame of all the files flagged for missing nodes
    """
    csv_name = "all_info_df.csv"
    bands = ["l", "s", "c", "x"]

    for band in bands:
        # gather csv files
        print("gathering %s band files..."%(band.upper()), end="")
        source_files = []
        csv_paths = glob.glob(data_dir + "%s-band/*"%band)
        for i in range(len(csv_paths)):
            filename = os.path.basename(csv_paths[i]).replace("dat", "h5")  # Need to fix the 
            source_files.append(filename)                                   # source file name tracking
            csv_paths[i] = csv_paths[i] + "/" + csv_name
        band_string = band*(len(csv_paths))
        band_list = list(band_string)
        
        print("Done.")
        # search the files
        missing_files_df = check_many_energy_detection_files(missing_files_df, csv_paths, source_files, band_list, threshold=threshold)
    
    return missing_files_df

def turbo_seti_driver(missing_files_df, data_dir, band):
    """
    function that checks many files and makes a list
    of which files suffer from data loss due to dropped nodes

    Arguments
    ----------
    missing_files_df : pandas.core.frame.DataFrame
        DataFrame of known files missing nodes. If no files
        are known, pass in an empty DataFrame
    data_dir : str
        path to the directory containing the folders
        with the results of running the energy detection 
        algorithm on many files
    band : str
        the band at which the data was collected
        choose from {"L", "S", "C", "X"}

    Returns
    --------
    missing_files_df : pandas.core.frame.DataFrame
        DataFrame of all the files flagged for missing nodes
    """
    print("gathering %s band files"%(band.upper()))
    source_files = []
    dat_files = glob.glob(data_dir + "/*dat")
    for i in range(len(dat_files)):
        filename = os.path.basename(dat_files[i]).replace("datnew.dat", "dat").replace("dat", "h5")
        source_files.append(filename)
    band_string = band*len(dat_files)
    band_list = list(band_string)

    missing_files_df = check_many_turbo_seti_files(missing_files_df, dat_files, source_files, band_list)
    
    return missing_files_df

def z_score(test_df, historical_df):
    """
    calculates the z-score for each entry in test_df 
    based on the corresponding data from historical_df

    Arguments
    ----------
    test_df : pandas.core.frame.DataFrame
        DataFrame of the test data
    historical_df : pandas.core.frame.DataFrame
        DataFrame of the historical dat files that 
        the test data will be compared to

    Returns
    --------
    z_tbl : numpy.ndarray
        the z-score of each entry in test_df, based
        on the data from historical_df
    """
    x = np.array(test_df)
    historical = np.array(historical_df)
    mean = np.mean(historical, axis=0)
    sd = np.std(historical, axis=0)
    no_data = np.where(np.sum(historical, axis=0) == 0)
    sd[no_data] = np.inf
    mask = np.where(sd == 0)
    sd[mask] = 1e-9
    z_tbl = (x - mean)/ sd
    
    return z_tbl

def remove_DC_spikes(dat_file, GBT_band, outdir=""):
    """
    removes the DC spike channels from a dat file 
    and stores its contents in a DataFrame

    Arguments
    ----------
    dat_file : str
        filepath to the dat file to be cleaned
    GBT_band : str
        the band at which the data was collected
        choose from {"L", "S", "C", "X"}
    outdir : str
        the filepath to where the cleaned dat files will be stored

    Returns
    --------
    df : pandas.core.frame.DataFrame
        DataFrame containing the cleaned dat file data
    """
    dc.remove_DC_spike(dat_file, outdir, GBT_band)
    filename = os.path.basename(dat_file)
    df = fe.read_dat(outdir + "/" + filename +"new.dat")
    return df

def check_dir(outdir):
    """
    checks for the existence of a directory and creates 
    the directory if it does not exist

    Arguments
    ----------
    outdir : str
        the directory path to be checked for
    """
    if not os.path.exists(outdir):
        os.mkdir(outdir)

def list_to_df(test_data_lst, band, outdir=""):
    """
    takes a list of dat file paths, removes their
    DC spike channels, and returns a DataFrame containing
    the cleaned data

    Arguments
    ----------
    test_data_lst : list
    band : str
        the band at which the data was collected
        choose from {"L", "S", "C", "X"}
    outdir : str
        the filepath to where the cleaned dat files will be temporarily stored

    Returns
    --------
    hist_df : pandas.core.frame.DataFrame
        DataFrame containing the histogram of the hit distribuiton
        from the set of dat files
    loc_df : pandas.core.frame.DataFrame
        DataFrame containing the RA, DEC, and MJD of the observation
        this can be used to calculate the pointing of the telescope
    """
    save_dats_to = outdir + "%s_band_no_DC_spike"%band
    check_dir(save_dats_to)
    df = pd.DataFrame()
    file_name = os.path.basename(test_data_lst[0])

    # remove DC spike channels and add first file histogram to dataframe
    no_DC_spikes = remove_DC_spikes(test_data_lst[0], GBT_band=band, outdir=save_dats_to)
    RA = no_DC_spikes["RA"]
    DEC = no_DC_spikes["DEC"]
    MJD = no_DC_spikes["MJD"]
    hist, bin_edges = so.calculate_hist(no_DC_spikes, band, bin_width=1)
    temp_dict = {"filename":file_name}
    for i in range(len(hist)):
        temp_dict[bin_edges[i]] = hist[i]
    df = df.append(temp_dict, ignore_index=True)
    loc_dict = {"filename":file_name, "RA":RA, "DEC":DEC, "MJD":MJD}
    loc_df = pd.DataFrame(loc_dict)
    
    # remove DC spike channels and add remaining histograms to dataframe
    for i in trange(len(test_data_lst)-1):
        file_name = os.path.basename(test_data_lst[i+1])
        no_DC_spikes = remove_DC_spikes(test_data_lst[i+1], GBT_band=band, outdir=save_dats_to)
        RA = no_DC_spikes["RA"]
        DEC = no_DC_spikes["DEC"]
        MJD = no_DC_spikes["MJD"]
        temp_loc_dict = {"filename":file_name, "RA":RA, "DEC":DEC, "MJD":MJD}
        temp_loc_df = pd.DataFrame(temp_loc_dict)
        loc_df = loc_df.append(temp_loc_df, ignore_index=True)
        hist, edges = so.calculate_hist(no_DC_spikes, band, bin_width=1)
        temp_df = pd.DataFrame()
        temp_dict = {"filename":file_name}
        for j in range(len(hist)):
            temp_dict[bin_edges[j]] = hist[j]
        temp_df = temp_df.append(temp_dict, ignore_index=True)
        df = df.append(temp_df, ignore_index=True)
    
    df = df.set_index("filename")

    # remove the directory containing the cleaned dat files
    remove_statement = "rm -rf %s"%save_dats_to
    os.system(remove_statement)

    return df, loc_df

def flag_z(test_df, historical_df, min_z):
    """
    Takes a DataFrame and minimum Z-score and returns 
    all the files with channels above this Z-score
    as well as the corresponding frequencies
    
    Arguments:
    -----------
    test_df : pandas.core.frame.DataFrame
        DataFrame containing the detected hit data
    historical_df : pandas.core.frame.DataFrame
        DataFrame containing historical data calculated from 
        long-term TESS targets observed at GBT
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
    flagged_max_z : numpy.ndarray
        Array of the higest z-score for a given file
    flagged_max_z_frequency : type(max_z_frequency)
        Array of the frequency(s) corresponding to the 
        highest z-score in a given file
    """
    z_tbl = z_score(test_df, historical_df)
    mask = np.where(z_tbl > min_z)
    stacked = np.vstack(mask)
    max_z = np.max(z_tbl, axis=1)
    max_z_frequencies = []
    frequencies = test_df.columns
    for i in range(len(test_df)):
        max_z_index = np.where(z_tbl[i] == max_z[i])
        max_z_frequencies.append(frequencies[max_z_index])
    file_indx = np.unique(stacked[0])
    flagged_freqs = []
    for indx in file_indx:
        this_file = stacked[1][np.where(stacked[0] == indx)[0]]
        frequencies = np.array(test_df.T.iloc[this_file].index)
        flagged_freqs.append(frequencies)
    flagged_files = test_df.iloc[file_indx].index
    flagged_max_z = max_z[file_indx]
    flagged_max_z_frequency = np.array(max_z_frequencies, dtype=object)[file_indx]
    return flagged_files, flagged_freqs, flagged_max_z, flagged_max_z_frequency

def gaussian(x, A, mu, sigma):
    """
    Returns a gaussian of the form
    A*exp(-0.5(mu - x)^2 / sigma^2)
    """
    return A*np.exp(-0.5 * (x-mu)**2 / sigma**2)

def set_threshold(mean, SD, threshold_SD, x_data, y_data):
    """
    sets the critical threshold for the number of 
    bins flagged for a file to be considered affected by RFI

    Arguments
    ----------
    mean : float
        the mean of the gaussian fit
    SD : float
        the standard deviation of the gaussian fit
    threshold_SD : float
        the number of standard deviations above the mean 
        the number of hits must be for it to be flagged
    x_data : numpy.ndarray
        the bin edges of the corresponding histogram
    y_data : numpy.ndarray
        the bin heights of the corresponding histogram

    Returns
    --------
    critical_x : float
        the threshold value for the number of flagged bins, 
        above which a file will be considered to be affected by RFI
    """
    critical_x = (mean + threshold_SD*SD)
    mask = np.where(x_data[:-1] > critical_x)
    above_threshold = y_data[mask]
    return critical_x

def percent_above_threshold(threshold, x_data, y_data):
    """
    calculates the percent of files above the determined threshold

    Arguments
    ----------
    threshold : float
        the threshold above which files will be considered flagged
    x_data : numpy.ndarray
        the bin edges of the histogram
    y_data : numpy.ndarray
        the bin heights of the histogram

    Returns
    --------
    percent_flagged : float
        the percent of files found to be above the threshold
    """
    mask = np.where((x_data[:-1] > threshold) & (y_data > 0))
    return np.sum(y_data[mask])/np.sum(y_data)

def identify_flagged_files(threshold, filenames, freqs, max_z, max_z_frequency, band):
    """
    identifies the name, band, bins flagged, max z score, and frequency of the highest z-score(s)

    Arguments
    ----------
    threshold : float
        the threshold above which files will be considered flagged
    filenames : list
        the names of the files that were flagged for having 
        higher than usual hit counts in some bins
    freqs : numpy.ndarray
        the frequencies that were flagged as having 
        a higher than usual number of hits
    max_z : numpy.ndarray
        the highest z_score for a given file
    max_z_frequency : numpy.ndarray
        the frequency(s) corresponding with the highest z-score
    band : str
        the band at which the data was collected
        choose from {"L", "S", "C", "X"}

    Returns
    --------
    flag_df : pandas.core.frame.DataFrame
        DataFrame containing the relevant info for the flagged files.
        Example row from DataFrame
                    filename            band        bins flagged          max z score         max z frequency bin
                    example file.h5      L              42                  12.2245                 1234
    """
    if type(filenames) == pandas.core.indexes.base.Index:
        filenames = filenames.values
    n_flags = np.empty(len(freqs))
    for i in range(len(freqs)):
        n_flags[i] = len(freqs[i])
        temp = filenames[i].replace("datnew.dat", "dat")
        temp = temp.replace("dat", "h5")
        filenames[i] = temp
    
    mask = (n_flags > threshold)
    out_files = filenames[mask]
    out_flags = n_flags[mask]
    out_z_scores = max_z[mask]
    out_z_frequency = max_z_frequency[mask]
    new_out_z_freq = []
    for i in range(len(out_z_frequency)):
        if len(out_z_frequency[i]) > 1:
            new_out_z_freq.append(' '.join(str(v) for v in out_z_frequency[i]))
        else:
            new_out_z_freq.append(str(out_z_frequency[i][0]))

    # target_names = []
    # for target in out_files:
    #     target_names.append(get_target_name(target))
    
    data_dict = {"filename":filenames[mask], "band":[band]*np.sum(mask), "bins flagged":n_flags[mask], "max z score":out_z_scores, "max z frequency bin":new_out_z_freq}
    return pd.DataFrame(data_dict)

def RFI_check(test_df, out_dir, GBT_band, sigma_threshold=2, bad_file_threshold=5, algorithm="turboSETI"):
    """
    checks dat files against historical data to determine 
    whether or not they were affected by RFI

    Arguments
    ----------
    test_df : pandas.core.frame.DataFrame
        DataFrame containing the detected event data
    out_dir : str
        the filepath to where the results will be stored
    GBT_band : str
    the band at which the data was collected
        choose from {"L", "S", "C", "X"}
    sigma_threshold : float
        the number of standard deviations from the mean
        a hit count needs to be in order to be flagged
    bad_file_threshold : float
        the number of standard deviations from the historical
        mean flag count distribution a file needs to be in order 
        to be considered affected by RFI
    algorithm : str
        algorithm used to generate the files. 
        Chose from {turboSETI, energy_detection} as input

    Returns
    --------
    bad_file_df : pandas.core.frame.DataFrame
        DataFrame containing information about the files
        that are flagged for being affected by RFI
    band_flag_count.pdf : saved file
        a plot showing the distribution of the test files 
        compared to the gaussian fit to the historical data
    """
    percent = "%"

    if algorithm == "turboSETI":
        import TESS_statistics.turboSETI.get_data as gd
        historical_data = gd.get_data()
    elif algorithm == "energy_detection":
        import TESS_statistics.energy_detection.get_data as gd
        historical_data = gd.get_data()
    else:
        print("ERROR:: please enter an acceptable algorithm input from the following:")
        print("{turboSETI, energy_detection}")
        exit()

    dat_list = glob.glob(test_df+"/*dat")
    test_df, test_loc_df = list_to_df(dat_list, GBT_band, outdir=out_dir)

    plt.figure(figsize=(6,6))
    bad_file_df = pd.DataFrame()

    # read in long-term data
    historical_df = historical_data[GBT_band]
    min_f, max_f = so.band_edges(GBT_band)
    freqs = np.arange(min_f, max_f)

    # exclude bandpass data for L,S band sources
    if GBT_band == "L":
        mask = np.where((freqs >=1200) & (freqs<= 1341))
    elif GBT_band == "S":
        mask = np.where((freqs >= 2300) & (freqs <= 2360))
    else:
        mask = np.where((freqs >= max_f) & (freqs <= min_f))
    band_data = test_df.to_numpy()
    historical = historical_df.to_numpy()
    band_data[:, mask] = 0
    historical[:, mask] = 0

    # identify files with high z-scores
    a, b, max_z, max_z_frequency = flag_z(test_df, historical_df, sigma_threshold)
    flag_counts = []
    for entry in b:
        flag_counts.append(len(entry))

    # set cutoff based on long-term statistics
    hist_files, hist_freqs, hist_max_z, hist_max_z_frequency = flag_z(historical_df, historical_df, sigma_threshold)
    hist_counts = []
    for entry in hist_freqs:
        hist_counts.append(len(entry))
    
    # set bins for specific band
    if GBT_band == "L":
        bins=50
    elif GBT_band == "S":
        bins = np.arange(0, 41, 2)
    elif GBT_band == "C":
        bins = np.arange(0, 151, 3)
    elif GBT_band == "X":
        bins = np.arange(0, 151, 7)

    hy, hx = np.histogram(hist_counts, bins=bins)
    popt, pcov = curve_fit(gaussian,hx[:-1],hy)#, p0=(3, 200))
    critical_x = set_threshold(popt[1], popt[2], bad_file_threshold, hx, hy)

    # rescale amplitude of gaussian
    popt[0] = popt[0]/len(historical_df)*len(test_df)

    # make histogram of test data
    y,x,_ = plt.hist(flag_counts, bins=50)
    plt.vlines(critical_x, 0, popt[0], color="red")
    plot_x = np.linspace(0, np.max(x), 1000)
    flag_fraction = percent_above_threshold(critical_x, x, y)
    bad_file_df = bad_file_df.append(identify_flagged_files(critical_x, a, b, max_z, max_z_frequency, GBT_band), ignore_index=True)
    plt.scatter(0,0, s=1, label="percent bad files: %.2f%s"%((flag_fraction*100), percent))
    plt.plot(plot_x, gaussian(plot_x, *popt), label="fit $\sigma$ = %.3f"%popt[-1])
    plt.legend()
    plt.title("L band channels above %s$\sigma$"%sigma_threshold)
    plt.xlabel("Number of channels flagged")
    plt.ylabel("count")

    plt.savefig(out_dir + "%s_band_flag_count.pdf"%GBT_band, bbox_inches="tight", transparent=False)

    return bad_file_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="passes over turboSETI or Energy Detection files and identifies files that are affected by compute node dropout, identifiable by a total absense of data in an interval of 187.5 MHz from the start of the channel")
    parser.add_argument("band", help="the GBT band that the data was collected from. Either L, S, C, or X")
    parser.add_argument("data_dir", help="directory where the output files (.dat or .csv) are stored")
    parser.add_argument("algorithm", help="algorithm used to generate the files. Choose from {turboSETI, energy_detection} as input")
    parser.add_argument("-threshold", "-t", help="Note: for energy detection files only! The threshold below which all hits will be excluded. Default is 4096", type=float, default=4096)
    parser.add_argument("-outdir", "-o", help="directory where the results are saved", default="")
    args = parser.parse_args()

    # fix potential pathing error
    if args.outdir != "":
        args.outdir = args.outdir + "/"

    missing_files_df = pd.DataFrame()

    if args.algorithm == "energy_detection":
        missing_files_df = energy_detection_driver(missing_files_df, args.data_dir, threshold=args.threshold)
        missing_files_df.to_csv(args.outdir + "%s_band_dropped_nodes.csv"%args.band, index=False)
    elif args.algorithm == "turboSETI":
        print("Checking for dropped nodes")
        missing_files_df = turbo_seti_driver(missing_files_df, args.data_dir, args.band)
        missing_files_df.to_csv(args.outdir + "%s_band_dropped_nodes.csv"%args.band, index=False)
        print("Checking for high RFI")
        RFI_df = RFI_check(test_df=args.data_dir, out_dir=args.outdir, GBT_band=args.band, algorithm=args.algorithm)
        RFI_df.to_csv(args.outdir + "%s_band_bad_RFI.csv"%args.band, index=False)
    else:
        print("ERROR:: please enter an acceptable algorithm input from the following:")
        print("{turboSETI, energy_detection}")
        exit()

    print("Done!")