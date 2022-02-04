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
    
    means = np.empty_like(boundaries)
    st_devs=np.empty_like(boundaries)

    for i in range(len(boundaries)):
        fch1 = boundaries[i]
        df_subset = select_node(df, fch1)
        bins = np.linspace(fch1-187.5, fch1, num=1875, endpoint=True)
        hist, bin_edges = np.histogram(df_subset["frequency"].values, bins=bins)
        means[i] = np.mean(hist)
        st_devs[i] = np.std(hist)

    return means, st_devs

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

if __name__ == "__main__":
    df = pd.read_csv("/Users/DanielBautista/Research/data/energy-detection/spliced_blc5051525354555657_guppi_58892_35102_HIP53639_0025/all_info_df.csv")
    df.rename(columns={"freqs":"frequency"}, inplace=True)
    nodes, boundaries = node_boundaries("L")
    means, sds = boxcar_analysis(df, nodes, boundaries)
    print(means)
    print(sds)