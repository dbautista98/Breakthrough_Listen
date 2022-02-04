from tabnanny import check
import pandas as pd
import numpy as np

def node_boundaries(band):
    """
    returns the compute nodes and their boundaries for an observing band

    Arguments
    ----------
    band : str
        the band at which the data was collected
        choose from {"L", "S", "C", "X"}
    
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

def boxcar_analysis(df, nodes, boundaries):
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
        bins = np.linspace(fch1-187.5, fch1, num=1875, endpoint=True)
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

    return result_df#means, st_devs

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

def check_missing(results_df):
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
        a node has been dropped during analysis. A (1) indicates that the 
        node was dropped and a (0) indicates that the node was included and
        that nothing went wrong 
    nodes : list
        the name of the compute node which the data was processed on
    
    Returns
    --------
    node_list : list
        a list of all the nodes that were dropped during data recording
    """
    bit_list = np.array(list(bitmap))
    mask = (bit_list == "1")
    return list(np.array(nodes)[mask])

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
    results = boxcar_analysis(df, nodes, boundaries)
    missing_string = check_missing(results)
    dropped_node_list = identify_missing_node(missing_string, nodes)
    if len(dropped_node_list) == 0:
        # return an empty DataFrame
        return pd.DataFrame()
    summary_dict = {"filename":source_file_name, "band":band.upper(), "dropped node bitmap":missing_string, "algorithm":"energy detection"}
    temp_df = pd.DataFrame()
    summary_df = temp_df.append(summary_dict, ignore_index=True)
    return summary_df

if __name__ == "__main__":
    ask_user = input("Is this running in GCP? y/n\n")
    if ask_user == "n":
        full_run = input("Do you want to run full program? y/n\n")
        if full_run == "y":
            node_summary = energy_detection_file_summary("/Users/DanielBautista/Research/data/energy-detection/spliced_blc5051525354555657_guppi_58892_35102_HIP53639_0025/all_info_df.csv", 
                                                        "L", "this is only a test.h5")
            print(node_summary)
        else:
            df = pd.read_csv("/Users/DanielBautista/Research/data/energy-detection/spliced_blc5051525354555657_guppi_58892_35102_HIP53639_0025/all_info_df.csv")
            df = format_energy_detection(df)
            nodes, boundaries = node_boundaries("L")
            results = boxcar_analysis(df, nodes, boundaries)
            print(results)
            missing_string = check_missing(results)
            print("Missing nodes:", missing_string)
            print(identify_missing_node(missing_string, nodes))
            # print(identify_missing_node("01010001", nodes))
    else:
        full_run = input("Do you want to run full program? y/n\n")
        if full_run == "y":
            node_summary = energy_detection_file_summary('/home/dbautista98/energy-detection/x-band/spliced_blc00010203040506o7o0111213141516o0212224252627_guppi_58806_44811_HIP68589_0132/all_info_df.csv', 
                                                        "X", "this is only a test.h5")
            print(node_summary)
        else:
            df = pd.read_csv('/home/dbautista98/energy-detection/x-band/spliced_blc00010203040506o7o0111213141516o0212224252627_guppi_58806_44811_HIP68589_0132/all_info_df.csv')
            nodes, boundaries = node_boundaries("X")
            df = format_energy_detection(df)
            results = boxcar_analysis(df, nodes, boundaries)
            print(results)
            missing_string = check_missing(results)
            print("Missing nodes:", missing_string)
            print(identify_missing_node(missing_string, nodes))
            # print(identify_missing_node("01010001", nodes))