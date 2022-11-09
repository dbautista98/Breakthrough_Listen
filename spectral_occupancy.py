import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas
import glob
import argparse
import os
from tqdm import trange
import pickle
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astroplan import Observer
from astroplan.plots import plot_sky
import re

figsize = (10,5)

def band_edges(GBT_band):
    """
    Returns the edge frequencies of the Green Bank Telescope
    bands for {L, S, C, X} bands, as listen in Traas 2021

    Arguments
    ----------
    GBT_band : str
        the band at which the data was collected
        choose from {"L", "S", "C", "X"}
    
    Returns
    --------
    min_freq : float
        lowest frequency in the bandpass
    max_freq : float
        highest frequency in the bandpass
    """
    if GBT_band.upper()=="L":
        min_freq = 1100
        max_freq = 1900
    if GBT_band.upper()=="S":
        min_freq = 1800
        max_freq = 2800
    if GBT_band.upper()=="C":  
        min_freq = 4000
        max_freq = 7800
    if GBT_band.upper()=="X":
        min_freq = 7800
        max_freq = 11200
    return min_freq, max_freq

def remove_spikes(dat_files, GBT_band, outdir="."):
    """
    Calls DC spike removal code on the list of 
    .dat files. Reads a .dat file and generates
    and saves a new .dat file that has no DC spikes 
    
    Arguments
    ----------
    dat_files : lst
        A python list containing the filepaths of 
        all the dat files which will have their 
        DC spikes removed
    GBT_band : str
        the band at which the data was collected
        choose from {"L", "S", "C", "X"}
    outdir : str
        directory where the results are saved
        
    Returns
    ----------
    new_dat_files : lst
        a python list of the filepaths to the new
        .dat files which no longer contain DC spikes
    """
    import remove_DC_spike
    
    new_dat_files = []

    # determine where to save new file
    checkpath = outdir + "/%s_band_no_DC_spike"%GBT_band
    if os.path.isdir(checkpath):
        pass
    else:
        os.mkdir(checkpath)

    # check if any of the files have been cleaned already
    to_clean = []
    for i in range(len(dat_files)):
        one_dat = os.path.basename(dat_files[0]) + "new.dat"
        one_path = checkpath + "/" + one_dat
        if not os.path.exists(one_path):
            to_clean.append(dat_files[0])
        else:
            new_dat_files.append(one_path)

    
    for i in trange(len(dat_files)):
        #get the path
        dat = dat_files[i]
        old_dat = os.path.basename(dat)
        
        remove_DC_spike.remove_DC_spike(dat, checkpath, GBT_band)
        
        newpath = checkpath + "/" + old_dat + "new.dat"
        new_dat_files.append(newpath)
    return new_dat_files

def read_txt(text_file):
    """
    reads a text file with one filepath per
    line and returns a python list where
    each entry is a filepath
    
    Arguments
    ----------
    text_file : str
        A string indicating the location of the 
        text file pointing to the dat files 

    Returns
    --------
    lines : list
        A list containing the lines of the txt file
    """
    with open(text_file) as open_file:
        lines = open_file.readlines()
    
    to_remove = []
    for i in range(len(lines)):
        lines[i] = lines[i].replace("\n", "")
        if "_" not in os.path.basename(lines[i]):
            to_remove.append(lines[i])
    n_remove = len(to_remove)
    if n_remove > 0:
        for i in range(n_remove):
            lines.remove(to_remove[i])
    return lines

def custom_read_dat(filename):
    """
    Read a turboseti .dat file.
    Arguments
    ----------
    filename : str
        Name of .dat file to open.

    Returns
    -------
    df_data : dict
        Pandas dataframe of hits.
    MJD : float
        the timestamp of the observation
    """
    file_dat = open(filename.strip())
    hits = file_dat.readlines()

    # Get info from the .dat file header
    FileID = hits[1].strip().split(':')[-1].strip()
    Source = hits[3].strip().split(':')[-1].strip()

    MJD = hits[4].strip().split('\t')[0].split(':')[-1].strip()
    mjd = hits[4].strip().split('\t')[0].split(':')[-1].strip()
    RA = hits[4].strip().split('\t')[1].split(':')[-1].strip()
    DEC = hits[4].strip().split('\t')[2].split(':')[-1].strip()

    DELTAT = hits[5].strip().split('\t')[0].split(':')[-1].strip()  # s
    DELTAF = hits[5].strip().split('\t')[1].split(':')[-1].strip()  # Hz

    # Get info from individual hits (the body of the .dat file)
    all_hits = []
    for hit_line in hits[9:]:
        hit_fields = re.split(r'\s+', re.sub(r'[\t]', ' ', hit_line).strip())
        all_hits.append(hit_fields)

    # Now reorganize that info to be grouped by column (parameter)
    # not row (individual hit)
    if all_hits:
        TopHitNum = list(zip(*all_hits))[0]
        DriftRate = [float(df) for df in list(zip(*all_hits))[1]]
        SNR = [float(ss) for ss in list(zip(*all_hits))[2]]
        Freq = [float(ff) for ff in list(zip(*all_hits))[3]]
        ChanIndx = list(zip(*all_hits))[5]
        FreqStart = list(zip(*all_hits))[6]
        FreqEnd = list(zip(*all_hits))[7]
        CoarseChanNum = list(zip(*all_hits))[10]
        FullNumHitsInRange = list(zip(*all_hits))[11]

        data = {'TopHitNum': TopHitNum,
                'DriftRate': DriftRate,
                'SNR': SNR,
                'Freq': Freq,
                'ChanIndx': ChanIndx,
                'FreqStart': FreqStart,
                'FreqEnd': FreqEnd,
                'CoarseChanNum': CoarseChanNum,
                'FullNumHitsInRange': FullNumHitsInRange
                }

        # Creating pandas dataframe from data we just read in
        df_data = pd.DataFrame(data)
        df_data = df_data.apply(pd.to_numeric)

    else:
        df_data = pd.DataFrame()

    # Matching column information from before to the .dat data we read in
    df_data['FileID'] = FileID
    df_data['Source'] = Source.upper()
    df_data['MJD'] = MJD
    df_data['RA'] = RA
    df_data['DEC'] = DEC
    df_data['DELTAT'] = DELTAT
    df_data['DELTAF'] = DELTAF

    # Adding extra columns that will be filled out by this program
    df_data['Hit_ID'] = ''
    df_data['status'] = ''
    df_data['in_n_ons'] = ''
    df_data['RFI_in_range'] = ''

    return df_data, float(mjd)

def calculate_hist(dat_file, GBT_band, bin_width=1, tbl=None): 
    """
    calculates a histogram of the number of hits for a single .dat file
    
    Arguments
    ----------
    dat_file : str
        filepath to the .dat file
    GBT_band : str
        the band at which the data was collected
        choose from {"L", "S", "C", "X"}
    bin_width : float
        width of the hisrogram bins in units of MHz
        The default is 1 Mhz
    tbl : pandas.core.frame.DataFrame
        Alternate way of providing contents of a dat file
        
    Returns
    --------
    hist : numpy.ndarray 
        the count of hits in each bin
    bin_edges : numpy.ndarray
        the edge values of each bin. Has length Nbins+1
    MJD : float
        the modified julian date at the start of the observation
    """
    #read the file into a pandas dataframe
    if type(dat_file) != pandas.core.frame.DataFrame:
        tbl, mjd = custom_read_dat(dat_file) # find.read_dat(dat_file)
    else:
        tbl = dat_file
        mjd = -9999

    #make the bins for the histogram
    # band boundaries as listed in Traas 2021
    if GBT_band=="L":
        min_freq = 1100
        max_freq = 1901
    if GBT_band=="S":
        min_freq = 1800
        max_freq = 2801
    if GBT_band=="C":
        min_freq = 4000
        max_freq = 7801
    if GBT_band=="X":
        min_freq = 7800
        max_freq = 11201
    bins = np.arange(min_freq, max_freq+0.5*bin_width, bin_width)#np.linspace(min_freq, max_freq, int((max_freq-min_freq)/bin_width) , endpoint=True)
    if len(tbl) == 0:
        hist, bin_edges = np.histogram([], bins=bins)
    else:
        hist, bin_edges = np.histogram(tbl["Freq"], bins=bins)
    return hist, bin_edges, mjd

def calculate_proportion(file_list, GBT_band, notch_filter=False, bin_width=1, outdir=".", title_addition=""):
    """
    Takes in a list of .dat files and makes a true/false table of hits in a frequency bin
    
    Arguments
    ----------
    file_list : list
        A python list containing the filepaths to .dat 
        files that will be used to calculate the 
        spcetral occupancy
    GBT_band : str
        the band at which the data was collected
        choose from {"L", "S", "C", "X"}
    notch_filter : bool
        A flag indicating whether or not to remove data 
        that fell within the notch filter. Note to user:
        only L and S band have notch filters
    bin_width : float
        width of the hisrogram bins in MHz
    outdir : str
        directory where the results are saved

    Returns
    --------
    bin_edges : numpy.ndarray
        the edge values of each bin. Has length Nbins+1
    occupancy : numpy.ndarray
        The spectral occupancy, defined as the fraction of observations
        in which at least one high SNR signal was detected by turboSETI
    n_observations : int
        The number of observations in the dataset
    """

    histograms = []
    
    # collect spliced and unspliced observations
    spliced_df, unspliced_df = spliced_unspliced_split(file_list)
    unique_unspliced_observations = get_unique_observations(unspliced_df)

    print("Calculating histograms...",end="")
    mjds = []
    # calculate histograms for the spliced .dat files
    for file in spliced_df["filepath"].values:
        hist, bin_edges, mjd = calculate_hist(file, GBT_band, bin_width)
        histograms.append(hist)
        mjds.append(float(mjd))

    # calculate the histograms for the unspliced .dat files
    bad_cadence_flag = False
    for obs in unique_unspliced_observations:
        good_nodes, bad_cadence_flag = getGoodNodes(obs, GBT_band, bad_cadence_flag, outdir=outdir, title_addition=title_addition)
        if type(good_nodes) == int:
            continue
        hist, bin_edges, mjd = calculate_hist(good_nodes[0], GBT_band, bin_width)
        for i in range(1, len(good_nodes)):
            temp_hist, bin_edges, mjd = calculate_hist(good_nodes[i], GBT_band, bin_width)
            hist += temp_hist
        histograms.append(hist)
        mjds.append(float(mjd))
    print("Done.")  

    # plot heatmap
    plot_heatmap(histograms, GBT_band, outdir=outdir, times=mjds, title_addition=title_addition)
    
    # define the upper and lower bounds of the band
    min_freq = np.min(bin_edges)
    max_freq = np.max(bin_edges)

    #create the dataframe and add the frequency bins to column 0
    df = pd.DataFrame()
    df.insert(0, "freq", bin_edges[:-1])
    
    #check if there is a hit in the frequency bin and insert value to dataframe
    for i in range(len(histograms)):
        colname = "file"+str(i)
        found_hit = histograms[i] > 0
        df.insert(len(df.columns), colname, found_hit.astype(int))
    
    #exclude entries in the GBT data due to the notch filter exclusion
    bin_edges = np.arange(min_freq, max_freq+0.5*bin_width, bin_width)#np.linspace(min_freq, max_freq, int((max_freq-min_freq)/bin_width), endpoint=True)
    if GBT_band=="L":
        if notch_filter:
            print("Excluding hits in the range 1200-1341 MHz")
            mask = np.where((df["freq"] > 1200) & (df["freq"] < 1341))
            for col in df.columns[1:]:
                df[col].values[mask] = 0
    
    if GBT_band=="S":
        if notch_filter:
            print("Excluding hits in the range 2300-2360 MHz")
            mask = np.where((df["freq"] > 2300) & (df["freq"] < 2360))
            for col in df.columns[1:]:
                df[col].values[mask] = 0
     
    # sum up the number of entries that have a hit and divide by the number of .dat files
    data_labels = df.columns[2:]
    total = df["file0"].values
    for label in data_labels:
        total = total + df[label].values
    
    return bin_edges, total/len(histograms), len(histograms)

def spliced_unspliced_split(dat_files):
    """
    separates list of dat files into DataFrames of
    spliced and unspliced dat files

    Arguments
    ----------
    dat_files : list
        list of filepaths to the dat files

    Returns
    --------
    spliced_df : pandas.core.frame.DataFrame
        DataFrame containing the filepaths of spliced dat files
        column names: {"filepath", "MJD", "spliced"}
    unspliced_df : pandas.core.frame.DataFrame
        DataFrame containing the filepaths of unspliced dat files
        column names: {"filepath", "MJD", "spliced"}
    """
    times = []
    spliced = []
    for i in range(len(dat_files)):
        filename = os.path.split(dat_files[i])[1]
        if filename[:5] == "guppi": # some unspliced files start with guppi_{numbers}_{numbers}*
            spliced.append(False)
            chopped = filename.split("_")
            times.append(float(chopped[1] + "." + chopped[2])) 
        elif filename.find("spliced") == -1: # these files are also not spliced
            spliced.append(False)
            chopped = filename.split("_")
            times.append(float(chopped[2] + "." + chopped[3]))
        else: # these are the spliced files 
            spliced.append(True)
            chopped = filename.split("_")
            times.append(float(chopped[3] + "." + chopped[4]))
    df =  pd.DataFrame({"filepath":dat_files, "MJD":times, "spliced":spliced})

    return df[df["spliced"] == True], df[df["spliced"] == False]

def get_unique_observations(df):
    """
    Identifies the unique files corresponding to an observation

    Arguments
    ----------
    df : pandas.core.frame.DataFrame
        DataFrame containing filepaths to .dat files 
        from many observations or compute nodes

    Returns
    --------
    unique_observations : list
        A list in which each entry is a list containing the filepaths of the 
        dat files that make up that observation
    """
    unique_times = np.unique(df["MJD"])
    unique_observations = []
    for i in range(len(unique_times)):
        temp_df = df[df["MJD"] == unique_times[i]]
        unique_names = []
        unique_paths = []
        for j in range(len(temp_df)):
            name = os.path.split(temp_df.iloc[j]["filepath"])[1]
            if name not in unique_names:
                unique_names.append(name)
                unique_paths.append(temp_df.iloc[j]["filepath"])
        unique_observations.append(unique_paths)
    return unique_observations

def record_bad_cadence(first_dat, band, nodes, outdir=".", alread_called=False, title_addition=""):
    """
    records metadata about a flagged bad cadence to a csv file

    Arguments
    ----------
    first_dat : str
        dat file for the first node in the observation
    band : str
        the band at which the data was collected
        choose from {"L", "S", "C", "X"}
    nodes : list
        list of the node names that are present
    outdir : str
        directory where the results are saved
    already_called : bool
        A flag aboout whether there has already been a bad cadence
        flagged in the dataset. Set to False by default. A False flag
        will cause a new csv header to be printed before writing the metadata. 
        A True flag will skip the writing of a new csv header, and will go
        straight to writing the metadata
    """
    if title_addition == "":
        csv_path = outdir + "/%sband_bad_cadences.csv"%(band.lower())
    else:
        csv_path = outdir + "/%sband_bad_cadences_%s.csv"%(band.lower(), title_addition.replace(" ", "_"))
    print("bad cadence")
    print("recording to: " + csv_path)
    uqnodes = sorted(np.unique(nodes))
    if not alread_called: # start new csv file
        with open(csv_path, "w") as f:
            # write csv header
            f.write("target,n_nodes,nodes_present\n") # csv header
    
    # write the rest of the data
    with open(csv_path, "a") as f:
        # get target name
        filename = os.path.basename(first_dat)
        if filename[:5] == "guppi":
            target = filename.split("_")[3]
            index = 3
        else:
            target = filename.split("_")[4]
            index = 4
        # handle edge cases :):
        odd_names = ["AND", "LGS", "SAG", "PSR", "WHI", "LV", "SEGUE", "CAS", "MCG", "KUG"]
        if target.upper() in odd_names:
            target = filename.split("_")[index] + "_" + filename.split("_")[index + 1]
        f.write(target.upper() + ",")

        # number of nodes present
        f.write(str(len(nodes)) + ",")

        # write nodes present
        for i in range(len(uqnodes)):
            f.write(uqnodes[i] + " ")
        f.write("\n")
    
def getGoodNodes(datfiles, band, bad_cadence_flag, outdir=".", title_addition=""):
    """
    Credit to Noah Franz for writing the original 
    algorithm in https://github.com/noahfranz13/BL-TESSsearch/blob/main/analysis/hit_analysis.ipynb

    Selects the nodes that will be used for analysis, and discards
    the overlap nodes

    Arguments
    ----------
    datfiles : list
        A list of the filepaths of unspliced dat files from a single observation
    band : str
        The observing band from Green Bank Telescope. Either {"L", "S", "C", "X"}
    bad_cadence_flag : bool
        A flag aboout whether there has already been a bad cadence
        flagged in the dataset
    outdir : str
        directory where the problematic cadences are saved

    Returns
    --------
    datfiles : list
        A list of the filepaths of unspliced dat files 
        that will be included in analysis, without the overlap nodes
    bad_cadence_flag : bool
        A flag aboout whether there has already been a bad cadence
        flagged in the dataset
    """
    def rm_nodes(nodes_to_rm, nodes):
        i_to_rm = []
        for node in nodes_to_rm:
            whereNodes = np.where(node == nodes)[0]
            i_to_rm.append(whereNodes)
        i_to_rm = np.array(i_to_rm).flatten()
        return i_to_rm


    datfiles = np.array(datfiles)
    if os.path.split(datfiles[0])[1].find('spliced') == -1:
        nodes = np.array([os.path.split(file)[1][:5] for file in datfiles])
        uqnodes = sorted(np.unique(nodes))
        if band == "L":
            if len(uqnodes) == 8:
                return datfiles, bad_cadence_flag
            else:
                # get node numbers
                node_string = ""
                for i in range(len(uqnodes)):
                    node_string = node_string + uqnodes[i][-1]
                okay_nodes = ["1234567", "0123456", "123456", "123467", "012346", "12346", "0123467"]
                if node_string in okay_nodes:
                    # this is ok and return the nodes
                    return datfiles, bad_cadence_flag
                else:
                    print(f'There are {len(nodes)} nodes, not 8 as L-Band requires')
                    record_bad_cadence(datfiles[0], band, nodes, outdir, bad_cadence_flag, title_addition=title_addition)
                    bad_cadence_flag = True
                    return -9999, bad_cadence_flag 
        elif band == 'S':
            # check if the files have 8 nodes
            if len(uqnodes) == 8:
                return datfiles, bad_cadence_flag
            else:
                # get node numbers
                node_string = ""
                for i in range(len(uqnodes)):
                    node_string = node_string + uqnodes[i][-1]
                okay_nodes = ["1234567", "0123456", "234567", "123456"]
                if node_string in okay_nodes:
                    # this is ok and return the nodes
                    return datfiles, bad_cadence_flag
                else:
                    print(f'There are {len(nodes)} nodes, not 8 as S-Band requires')
                    record_bad_cadence(datfiles[0], band, nodes, outdir, bad_cadence_flag, title_addition=title_addition)
                    bad_cadence_flag = True
                    return -9999, bad_cadence_flag
        elif band == 'C':
            if len(uqnodes) == 32:
                nodes_to_rm = [uqnodes[7], uqnodes[8], uqnodes[15], uqnodes[16], uqnodes[23], uqnodes[24]]
                i_to_rm = rm_nodes(nodes_to_rm, nodes)
                return np.delete(datfiles, i_to_rm), bad_cadence_flag
            else:
                # get node numbers
                node_string = ""
                for i in range(len(uqnodes)):
                    node_string = node_string + uqnodes[i][-1]
                if node_string == "12345670123456701234567012345":
                    nodes_to_rm = [uqnodes[6], uqnodes[7], uqnodes[14], uqnodes[15], uqnodes[22], uqnodes[23]]
                    i_to_rm = rm_nodes(nodes_to_rm, nodes)
                    return np.delete(datfiles, i_to_rm), bad_cadence_flag
                else:
                    print(f'There are {len(nodes)} nodes, not 32 as C-Band requires')
                    record_bad_cadence(datfiles[0], band, nodes, outdir, bad_cadence_flag, title_addition=title_addition)
                    bad_cadence_flag = True
                    return -9999, bad_cadence_flag
        elif band == 'X':
            if len(uqnodes) == 24:
                nodes_to_rm = [uqnodes[7], uqnodes[8], uqnodes[15], uqnodes[16]]
                i_to_rm = rm_nodes(nodes_to_rm, nodes)
                return np.delete(datfiles, i_to_rm), bad_cadence_flag
            else:
                # get node numbers
                node_string = ""
                for i in range(len(uqnodes)):
                    node_string = node_string + uqnodes[i][-1]
                if node_string == "01234567012345670123456":
                    nodes_to_rm = [uqnodes[7], uqnodes[8], uqnodes[15], uqnodes[16]]
                    i_to_rm = rm_nodes(nodes_to_rm, nodes)
                    return np.delete(datfiles, i_to_rm), bad_cadence_flag
                else:
                    print(f'There are {len(nodes)} nodes, not 24 as X-Band requires')
                    record_bad_cadence(datfiles[0], band, nodes, outdir, bad_cadence_flag, title_addition=title_addition)
                    bad_cadence_flag = True
                    return -9999, bad_cadence_flag
        else:
            raise ValueError(f'{band}-Band is not a valid band')
    else:
        print('This file is already spliced, returning all files')
        return datfiles, bad_cadence_flag

def plot_AltAz(df, plot_color="#1f77b4", label=""):
    """
    plots the altitude and azimuth of the given observations

    Arguments
    ----------
    df : pandas.core.frame.DataFrame
        DataFrame containing right ascension (RA), declinaiton (DEC)
        and modified julian date (MJD) values for each observation
    band : str
        the band at which the data was collected
        choose from {"L", "S", "C", "X"}
    outdir : str
        directory where the generated plots will be saved. 
        default is the current working directory
    """
    targets = SkyCoord(df["RA"].values, df["DEC"].values, unit=(u.hourangle, u.deg), frame="icrs")          
    times = Time(np.array(df["MJD"].values, dtype=float), format="mjd")
    GBT = Observer.at_site("GBT", timezone="US/Eastern")

    opacity = min(1, 2000/len(df))
    plot_sky(targets, GBT, times, style_kwargs={"s":2, "c":plot_color, "label":label, "alpha":opacity})

def get_AltAz(df):
    """
    Takes a dataframe containing the RA, DEC and MJD values 
    and returns the altitude and azimuth angles the telescope
    was pointing during the observation

    Arguments
    ----------
    df : pandas.core.frame.DataFrame
        DataFrame containing right ascension (RA), declinaiton (DEC)
        and modified julian date (MJD) values for each observation

    Returns
    --------
    ALT : astropy.coordinates.angles.Latitude
        The altidude angle of the telescope; how many degrees above the horizon
        the telescope was pointing
    AZ : astropy.coordinates.angles.Longitude
        the azimuthal angle the telescope was pointing; how many degrees from 
        true north it was aimed, following the right hand rule
    """
    targets = SkyCoord(df["RA"].values, df["DEC"].values, unit=(u.hourangle, u.deg), frame="icrs")          
    times = Time(np.array(df["MJD"].values, dtype=float), format="mjd") 
    gbt = EarthLocation(lat=38.4*u.deg, lon=-79.8*u.deg, height=808*u.m)
    gbt_altaz_transformer = AltAz(obstime=times, location=gbt)
    gbt_target_altaz = targets.transform_to(gbt_altaz_transformer)
    ALT = gbt_target_altaz.alt
    AZ = gbt_target_altaz.az

    return ALT, AZ

def plot_heatmap(hist, band, outdir=".", times=None, title_addition=""):
    """
    takes a set of histograms and plots them 
    as an image, with the horizontal axis showing
    the Frequency and the vertical showing the 
    order in time

    Arguments
    ----------
    hist : list
        list of histograms, where each entry is the 
        histogram of hits from a single observation
    band : str
        the band at which the data was collected
        choose from {"L", "S", "C", "X"}
    outdir : str
        directory where the results are saved
    times : list
        list of the start time of each observation
        in the list of histograms
    """
    
    # check to sort histograms by mjd
    if times is not None:
        temp = pd.DataFrame({"time":times, "data":hist})
        sorted = temp.sort_values(by="time")
        hist = sorted["data"].values
        hist = list(hist)
    
    hist = np.log10(np.asarray(hist)+1)

    if band == "L":
        plt.figure(figsize=(15,3))
        plt.imshow(hist, cmap="viridis_r", aspect="auto")
        plt.colorbar(label="log(hit count)")

        freqs = np.arange(1100, 1900.1)
        # ticks = np.linspace(1100, 1900, num=n_ticks)
        ticks = np.asarray([1100, 1200, 1300, 1380, 1420, 1500, 1600, 1700, 1800, 1900])
        indices = (np.where(np.in1d(freqs, ticks) == True)[0])
        plt.xticks(ticks=indices, labels=freqs[indices].astype(int))
        plt.xlabel("Frequency [MHz]")
        plt.title("L band hit counts %s"%title_addition)
        if title_addition != "":
            plt.savefig(outdir + "/L_band_heatmap%s.pdf"%("_"+title_addition.replace(" ", "_")), bbox_inches="tight", transparent=False)
        else:
            plt.savefig(outdir + "/L_band_heatmap%s.pdf"%(title_addition), bbox_inches="tight", transparent=False)
    
    if band ==  "S":
        plt.figure(figsize=(15,3))
        plt.imshow(hist, cmap="viridis_r", aspect="auto")
        plt.colorbar(label="log(hit count)")

        freqs = np.arange(1800, 2800.1)
        ticks = np.arange(1800, 2800.1, 100)
        indices = (np.where(np.in1d(freqs, ticks) == True)[0])
        plt.xticks(ticks=indices, labels=freqs[indices].astype(int))
        plt.xlabel("Frequency [MHz]")
        plt.title("S band hit counts %s"%title_addition)
        if title_addition != "":
            plt.savefig(outdir + "/S_band_heatmap%s.pdf"%("_"+title_addition.replace(" ", "_")), bbox_inches="tight", transparent=False)
        else:
            plt.savefig(outdir + "/S_band_heatmap%s.pdf"%(title_addition), bbox_inches="tight", transparent=False)
    
    if band == "C":
        plt.figure(figsize=(15,3))
        plt.imshow(hist, cmap="viridis_r", aspect="auto")
        plt.colorbar(label="log(hit count)")

        freqs = np.arange(4000, 7800.1)
        ticks = np.asarray([4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 7800])
        indices = (np.where(np.in1d(freqs, ticks) == True)[0])
        plt.xticks(ticks=indices, labels=freqs[indices].astype(int))
        plt.xlabel("Frequency [MHz]")
        plt.title("C band hit counts %s"%title_addition)
        if title_addition != "":
            plt.savefig(outdir + "/C_band_heatmap%s.pdf"%("_"+title_addition.replace(" ", "_")), bbox_inches="tight", transparent=False)
        else:
            plt.savefig(outdir + "/C_band_heatmap%s.pdf"%(title_addition), bbox_inches="tight", transparent=False)
    
    if band == "X":
        plt.figure(figsize=(15,3))
        plt.imshow(hist, cmap="viridis_r", aspect="auto")
        plt.colorbar(label="log(hit count)")

        freqs = np.arange(7800, 11200.1)
        ticks = np.append(np.arange(7800, 11200.1, 500), 11200)
        indices = (np.where(np.in1d(freqs, ticks) == True)[0])
        plt.xticks(ticks=indices, labels=freqs[indices].astype(int))
        plt.xlabel("Frequency [MHz]")
        plt.title("X band hit counts %s"%title_addition)
        if title_addition != "":
            plt.savefig(outdir + "/X_band_heatmap%s.pdf"%("_"+title_addition.replace(" ", "_")), bbox_inches="tight", transparent=False)
        else:
            plt.savefig(outdir + "/X_band_heatmap%s.pdf"%(title_addition), bbox_inches="tight", transparent=False)

def plot_ratio(on_hist, off_hist, bin_edges, band, outdir=".", title_addition=""):
    off_hist[off_hist == 0] = np.nan
    ratio = on_hist/off_hist

    plt.figure(figsize=figsize)
    plt.plot(bin_edges[:-1], ratio)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("ratio")
    plt.title("%s Band spectral occupancy ratio of %s / remaining"%(band, title_addition))
    plt.savefig(outdir + "/%s_band_ratio_%s.pdf"%(band, title_addition.replace(" ", "_")), bbox_inches="tight", transparent=False)

def split_data(band, df, on_mask, off_mask, outdir, width, notch_filter, save, lower, upper, split_type):
    on_df = df.iloc[on_mask]
    off_df = df.iloc[off_mask]
    on_title_addition = "%s %s to %s"%(split_type, lower, upper)
    off_title_addition = "%s %s to %s"%(split_type, upper, lower)
    on_color = "orange"
    off_color = '#1f77b4'

    plot_AltAz(on_df, plot_color=on_color, label=on_title_addition)
    plot_AltAz(off_df, plot_color=off_color, label=off_title_addition)
    plt.legend(loc="upper right", bbox_to_anchor=(0.3,0))
    plt.savefig(outdir + "/%s_band_GBT_alt_az_split_%s.pdf"%(band, on_title_addition.replace(" ", "_")), bbox_inches="tight", transparent=False)
    plt.close("all")

    bin_edges, on_prob_hist, n_observations_on = calculate_proportion(on_df["filepath"].values, bin_width=width, GBT_band=band, notch_filter=notch_filter, outdir=outdir, title_addition=on_title_addition)
    bin_edges, off_prob_hist, n_observations_off = calculate_proportion(off_df["filepath"].values, bin_width=width, GBT_band=band, notch_filter=notch_filter, outdir=outdir, title_addition=off_title_addition)

    plot_ratio(on_prob_hist, off_prob_hist, bin_edges, band=band, outdir=outdir, title_addition=on_title_addition)

    print("Saving plot...",end="")
    
    plt.figure(figsize=figsize)
    width = np.diff(bin_edges)[0]

    plt.plot(bin_edges[:-1], on_prob_hist, label=on_title_addition, linewidth=1, alpha=0.5, color=on_color)
    plt.plot(bin_edges[:-1], off_prob_hist, label=off_title_addition, linewidth=1, alpha=0.5, color=off_color)

    plt.xlabel("Frequency [Mhz]")
    plt.ylabel("Fraction with Hits")
    plt.title("%s Band Spectral Occupancy\nn=%s observations between %s\nn=%s observations between %s"%(band,n_observations_on, on_title_addition, n_observations_off, off_title_addition))
    plt.legend()
    plt.savefig(args.outdir + "/%s_band_spectral_occupancy_split_%s_%s_%s.pdf"%(band, split_type, lower, upper), bbox_inches="tight", transparent=False)
    plt.close("all")
    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generates a histogram of the spectral occupancy from a given set of .dat files")
    parser.add_argument("band", help="the GBT band that the data was collected from. Either L, S, C, or X")
    parser.add_argument("folder", help="directory .dat files are held in")
    parser.add_argument("-outdir", "-o", help="directory where the results are saved", default=os.getcwd())
    parser.add_argument("-t", help="a .txt file to read the filepaths of the .dat files", action=None)
    parser.add_argument("-width", "-w", help="width of bin in Mhz", type=float, default=1)
    parser.add_argument("-notch_filter", "-nf", help="exclude data that was collected within GBT's notch filter when generating the plot", action="store_true")
    parser.add_argument("-save", "-s", help="save the histogram bin edges and heights", action="store_true")
    parser.add_argument("-lower_time", "-lt", help="split the data into two time intervals. this sets the earlier time boundary. range is on 24 hour time", type=int, default=None)
    parser.add_argument("-upper_time", "-ut", help="split the data into two time intervals. this sets the earlier time boundary range is on 24 hour time", type=int, default=None)
    parser.add_argument("-lower_alt", "-la", help="split the data into two altitude intervals. this sets the lower altitude boundary. range is from 0-90 degrees", type=int, default=None)
    parser.add_argument("-upper_alt", "-ua", help="split the data into two altitude intervals. this sets the upper altitude boundary. range is from 0-90 degrees", type=int, default=None)
    parser.add_argument("-lower_az", "-lz", help="split the data into two azimuth intervals. this sets the lower azimuth boundary. range is from 0-360 degrees", type=int, default=None)
    parser.add_argument("-upper_az", "-uz", help="split the data into two azimuth intervals. this sets the upper azimuth boundary. range is from 0-360 degrees", type=int, default=None)
    parser.add_argument("-no_default", "-no", help="flag to tell program NOT to plot the default spectral occupancy with zero grouping of the data", default=False, action="store_true")
    args = parser.parse_args()
    
    print("Gathering files...",end="")
    if args.t == None:
        dat_files = glob.glob(args.folder+"/*_*.dat")
    else:
        dat_files = read_txt(args.t)
    print("Done.")
    
    # check to remove DC spikes
    if "%s_band_no_DC_spike"%args.band not in args.folder:
        print("Removing DC spikes...")
        dat_files = remove_spikes(dat_files, args.band, outdir=args.outdir)
        df = pd.read_csv(args.outdir + "/%s_band_no_DC_spike/locations.csv"%args.band)
    else:
        df = pd.read_csv(args.folder + "/locations.csv")

    title_addition = ""

    # check if splitting data by time
    if (args.lower_time is not None) and (args.upper_time is not None):
        print("splitting data between hours of %s and %s"%(args.lower_time, args.upper_time))
        on_mask = np.where((df["hour (UTC - 5)"] >= args.lower_time) & (df["hour (UTC - 5)"] <= args.upper_time))
        off_mask = np.where((df["hour (UTC - 5)"] < args.lower_time) | (df["hour (UTC - 5)"] > args.upper_time))
        split_data(args.band, df, on_mask, off_mask, args.outdir, args.width, args.notch_filter, args.save, args.lower_time, args.upper_time, split_type="hour")
    # check if splitting data by altitude
    if (args.lower_alt is not None) and (args.upper_alt is not None):
        print("splitting the data between the altitude angles of %s and %s degrees"%(args.lower_alt, args.upper_alt))
        on_mask = np.where((df["ALT"] >= args.lower_alt) & (df["ALT"] <= args.upper_alt))
        off_mask = np.where((df["ALT"] < args.lower_alt) | (df["ALT"] > args.upper_alt))
        split_data(args.band, df, on_mask, off_mask, args.outdir, args.width, args.notch_filter, args.save, args.lower_alt, args.upper_alt, split_type="altitude")
    # check if splitting data by azimuth
    if (args.lower_az is not None) and (args.upper_az is not None):
        print("splitting the data between the azimuth angles of %s and %s degrees"%(args.lower_az, args.upper_az))
        on_mask = np.where((df["AZ"] >= args.lower_az) & (df["AZ"] <= args.upper_az))
        off_mask = np.where((df["AZ"] < args.lower_az) | (df["AZ"] > args.upper_az))
        split_data(args.band, df, on_mask, off_mask, args.outdir, args.width, args.notch_filter, args.save, args.lower_az, args.upper_az, split_type="azimuth")
    # check if plotting in default manner
    if not args.no_default:
        plot_AltAz(df)
        plt.savefig(args.outdir + "/%s_band_GBT_alt_az.pdf"%(args.band), bbox_inches="tight", transparent=False)
        plt.close("all")

        bin_edges, prob_hist, n_observations = calculate_proportion(dat_files, bin_width=args.width, GBT_band=args.band, notch_filter=args.notch_filter, outdir=args.outdir)
        
        if args.save:
            print("Saving histogram data")
            to_save = {"bin_edges":bin_edges, "bin_heights":prob_hist, "band":args.band, "bin width":args.width, "algorithm":"turboSETI", "n files":len(dat_files)}
            filename = args.outdir + "/turboSETI_%s_band_spectral_occupancy_%s_MHz_bins.pkl"%(args.band, args.width)
            with open(filename, "wb") as f:
                pickle.dump(to_save, f)

        print("Saving plot...",end="")
        plt.figure(figsize=figsize)
        width = np.diff(bin_edges)[0]
        plt.bar(bin_edges[:-1], prob_hist, width=width)
        plt.xlabel("Frequency [Mhz]")
        plt.ylabel("Fraction with Hits")
        plt.title("Spectral Occupancy: n=%s observations"%n_observations)
        plt.savefig(args.outdir + "/%s_band_spectral_occupancy.pdf"%args.band, bbox_inches="tight", transparent=False)
        print("Done")