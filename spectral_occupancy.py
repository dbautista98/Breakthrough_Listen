from tempfile import tempdir
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas
import turbo_seti.find_event as find
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
    num_course_channels : int
        the number of course channels in a frequency band. The 
        default is 512
        
    Returns
    ----------
    new_dat_files : lst
        a python list of the filepaths to the new
        .dat files which no longer contain DC spikes
    loc_df : pandas.core.frame.DataFrame
        DataFrame containing the RA, DEC, and MJD of the observation
        this can be used to calculate the pointing of the telescope
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
        path = os.path.dirname(dat)
        old_dat = os.path.basename(dat)
        dat_data = find.read_dat(dat)
        RA = dat_data["RA"]
        DEC = dat_data["DEC"]
        MJD = dat_data["MJD"]
        
        remove_DC_spike.remove_DC_spike(dat, checkpath, GBT_band)
        
        newpath = checkpath + "/" + old_dat + "new.dat"
        new_dat_files.append(newpath)
        loc_dict = {"path":newpath, "RA":RA, "DEC":DEC, "MJD":MJD}
        temp_loc_df = pd.DataFrame(loc_dict)
        loc_df = loc_df.append(temp_loc_df, ignore_index=True)
    return new_dat_files, loc_df

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
    """
    with open(text_file) as open_file:
        lines = open_file.readlines()
    
    for i in range(len(lines)):
        lines[i] = lines[i].replace("\n", "")
    return lines

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
    """
    #read the file into a pandas dataframe
    if type(dat_file) != pandas.core.frame.DataFrame:
        tbl = find.read_dat(dat_file)
    else:
        tbl = dat_file

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
    return hist, bin_edges

def calculate_proportion(file_list, GBT_band, notch_filter=False, bin_width=1, outdir="."):
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
    edges = []
    histograms = []
    
    # collect spliced and unspliced observations
    spliced_df, unspliced_df = spliced_unspliced_split(file_list)
    unique_unspliced_observations = get_unique_observations(unspliced_df)

    print("Calculating histograms...",end="")
    # calculate histograms for the spliced .dat files
    for file in spliced_df["filepath"].values:
        hist, bin_edges = calculate_hist(file, GBT_band, bin_width)
        histograms.append(hist)

    # calculate the histograms for the unspliced .dat files
    bad_cadence_flag = False
    for obs in unique_unspliced_observations:
        good_nodes, bad_cadence_flag = getGoodNodes(obs, GBT_band, bad_cadence_flag, outdir=outdir)
        if type(good_nodes) == int:
            continue
        hist, bin_edges = calculate_hist(good_nodes[0], GBT_band, bin_width)
        for i in range(1, len(good_nodes)):
            temp_hist, bin_edges = calculate_hist(good_nodes[i], GBT_band, bin_width)
            hist += temp_hist
        histograms.append(hist)
    print("Done.")  
    
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
            df = df[(df["freq"] < 1200) | (df["freq"] > 1341)]
            first_edge = np.arange(min_freq, 1200, bin_width)
            second_edge= np.arange(1341, max_freq, bin_width) #may or may not need max_freq+1
            bin_edges = np.append(first_edge, second_edge)
    
    if GBT_band=="S":
        if notch_filter:
            print("Excluding hits in the range 2300-2360 MHz")
            df = df[(df["freq"] < 2300) | (df["freq"] > 2360)]
            first_edge = np.arange(min_freq, 2300, bin_width)
            second_edge= np.arange(2360, max_freq, bin_width) #may or may not need max_freq+1
            bin_edges = np.append(first_edge, second_edge)

     
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

def record_bad_cadence(first_dat, band, nodes, outdir, alread_called):
    csv_path = outdir + "/%sband_bad_cadences.csv"%band.lower()
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
        # handle edge cases (:(
        if target.upper() == "AND":
            target = filename.split("_")[index] + "_" + filename.split("_")[index + 1]
        if target.upper() == "LGS":
            target = filename.split("_")[index] + filename.split("_")[index + 1]
        if target.upper() == "SAG":
            target = filename.split("_")[index] + filename.split("_")[index + 1]
        f.write(target.upper() + ",")

        # number of nodes present
        f.write(str(len(nodes)) + ",")

        # write nodes present
        for i in range(len(uqnodes)):
            f.write(uqnodes[i] + " ")
        f.write("\n")
    
def getGoodNodes(datfiles, band, bad_cadence_flag, outdir="."):
    """
    Credit to Noah Franz for writing this 
    algorithm in https://github.com/noahfranz13/BL-TESSsearch/blob/main/analysis/hit_analysis.ipynb

    Selects the nodes that will be used for analysis, and discards
    the overlap nodes

    Arguments
    ----------
    datfiles : list
        A list of the filepaths of unspliced dat files from a single observation
    band : str
        The observing band from Green Bank Telescope. Either {"L", "S", "C", "X"}
    outdir : str
        directory where the problematic cadences are saved

    Returns
    --------
    datfiles : list
        A list of the filepaths of unspliced dat files 
        that will be included in analysis, without the overlap nodes
    """
    datfiles = np.array(datfiles)
    if os.path.split(datfiles[0])[1].find('spliced') == -1:
        nodes = np.array([os.path.split(file)[1][:5] for file in datfiles])
        if band == 'S' or band == 'L':
            # check if the files have 8 nodes
            if len(nodes) == 8:
                return datfiles, bad_cadence_flag
            else:
                print(f'There are {len(nodes)} nodes, not 8 as {band}-Band requires')
                record_bad_cadence(datfiles[0], band, nodes, outdir, bad_cadence_flag)
                bad_cadence_flag = True
                return -9999, bad_cadence_flag
        else:
            if band == 'C':
                if len(nodes) == 32:
                    uqnodes = sorted(np.unique(nodes))
                    nodes_to_rm = [uqnodes[7], uqnodes[8], uqnodes[15], uqnodes[16], uqnodes[23], uqnodes[24]]
                    i_to_rm = []
                    for node in nodes_to_rm:
                        whereNodes = np.where(node == nodes)[0]
                        i_to_rm.append(whereNodes)
                    i_to_rm = np.array(i_to_rm).flatten()
                    return np.delete(datfiles, i_to_rm), bad_cadence_flag
                else:
                    print(f'There are {len(nodes)} nodes, not 32 as C-Band requires')
                    record_bad_cadence(datfiles[0], band, nodes, outdir, bad_cadence_flag)
                    bad_cadence_flag = True
                    return -9999, bad_cadence_flag
            if band == 'X':
                if len(nodes) == 24:
                    uqnodes = sorted(np.unique(nodes))
                    nodes_to_rm = [uqnodes[7], uqnodes[8], uqnodes[15], uqnodes[16]]
                    i_to_rm = []
                    for node in nodes_to_rm:
                        whereNodes = np.where(node == nodes)[0]
                        i_to_rm.append(whereNodes)
                    i_to_rm = np.array(i_to_rm).flatten()
                    return np.delete(datfiles, i_to_rm), bad_cadence_flag
                else:
                    print(f'There are {len(nodes)} nodes, not 24 as X-Band requires')
                    record_bad_cadence(datfiles[0], band, nodes, outdir, bad_cadence_flag)
                    bad_cadence_flag = True
                    return -9999, bad_cadence_flag
    else:
        print('This file is already spliced, returning all files')
        return datfiles, bad_cadence_flag

def get_AltAz(loc_df, band, outdir="."):
    targets = SkyCoord(loc_df["RA"].values, loc_df["DEC"].values, unit=(u.hourangle, u.deg), frame="icrs")          
    times = Time(np.array(loc_df["MJD"].values, dtype=float), format="mjd") 
    gbt = EarthLocation(lat=38.4*u.deg, lon=-79.8*u.deg, height=808*u.m)
    gbt_altaz_transformer = AltAz(obstime=times, location=gbt)
    gbt_target_altaz = targets.transform_to(gbt_altaz_transformer)
    ALT = gbt_target_altaz.alt
    AZ = gbt_target_altaz.az
    loc_df["ALT"] = ALT
    loc_df["AZ"] = AZ

    # GBT = Observer.at_site("GBT", timezone="US/Eastern")
    # plot_sky(targets, GBT, times, style_kwargs={"c":"k"})
    # plt.savefig(outdir + "/%s_band_GBT_alt_az.png"%band, bbox_inches="tight", transparent=False)
    # plt.close()

    return loc_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generates a histogram of the spectral occupancy from a given set of .dat files")
    parser.add_argument("band", help="the GBT band that the data was collected from. Either L, S, C, or X")
    parser.add_argument("folder", help="directory .dat files are held in")
    parser.add_argument("-outdir", "-o", help="directory where the results are saved", default=os.getcwd())
    parser.add_argument("-t", help="a .txt file to read the filepaths of the .dat files", action=None)
    parser.add_argument("-width", "-w", help="width of bin in Mhz", type=float, default=1)
    parser.add_argument("-notch_filter", "-nf", help="exclude data that was collected within GBT's notch filter when generating the plot", action="store_true")
    # parser.add_argument("-DC", "-d", help="files contain DC spikes that need to be removed", action="store_true")
    parser.add_argument("-save", "-s", help="save the histogram bin edges and heights", action="store_true")
    parser.add_argument("-altitude_bins", "-a", help="number of degrees in an altitude bin, default is 90 degrees (the whole sky)", default=90)
    args = parser.parse_args()
    
    print("Gathering files...",end="")
    if args.t == None:
        dat_files = glob.glob(args.folder+"/*.dat")
    else:
        dat_files = read_txt(args.t)
    print("Done.")
    
    # check for argument to remove DC spikes
    # and identify the alt/az of the targets during observation
    # if args.DC:
    print("Removing DC spikes...")
    dat_files, loc_df = remove_spikes(dat_files, args.band)

    ## DEBUG::
    import time
    tstart = time.time()
    loc_df = get_AltAz(loc_df, args.band, outdir=args.outdir)
    print("Alt/Az conversion runtime: %s seconds"%(time.time() - tstart))
    print("Done.")
    if args.DC:
        print("Removing DC spikes...")
        dat_files = remove_spikes(dat_files, args.band, outdir=args.outdir)
        print("Done.")
    
    bin_edges, prob_hist, n_observations = calculate_proportion(dat_files, bin_width=args.width, GBT_band=args.band, notch_filter=args.notch_filter, outdir=args.outdir)
    
    if args.save:
        print("Saving histogram data")
        to_save = {"bin_edges":bin_edges, "bin_heights":prob_hist, "band":args.band, "bin width":args.width, "algorithm":"turboSETI", "n files":len(dat_files)}
        filename = args.outdir + "/turboSETI_%s_band_spectral_occupancy_%s_MHz_bins.pkl"%(args.band, args.width)
        with open(filename, "wb") as f:
            pickle.dump(to_save, f)

    print("Saving plot...",end="")
    plt.figure(figsize=(10,5))
    width = np.diff(bin_edges)[0]
    plt.bar(bin_edges[:-1], prob_hist, width=width)
    plt.xlabel("Frequency [Mhz]")
    plt.ylabel("Fraction with Hits")
    plt.title("Spectral Occupancy: n=%s"%n_observations)
    plt.savefig(args.outdir + "/%s_band_spectral_occupancy.pdf"%args.band, bbox_inches="tight", transparent=False)
    print("Done")