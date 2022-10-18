"""
Thanks to Karen Perez for sharing her code
to remove the DC spikes from a .dat file. 
"""

import numpy as np
import pandas as pd
import pandas
import glob
import argparse
import time
import turbo_seti.find_event as find
import os
from tqdm import trange

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

def grab_parameters(dat_file, GBT_band):
    """
    takes dat file of GBT data and returns frequency parameters 
    used to calculate where the DC spikes will be 

    Arguments
    ----------
    dat_file : str
        filepath to the .dat file
    GBT_band : str
        the band at which the data was collected
        choose from {"L", "S", "C", "X"}
        
    Returns : fch1, foff, nfpc
    which will be used internally within remove_DC_spike
    """
    if type(dat_file) != pandas.core.frame.DataFrame:
        tbl = find.read_dat(dat_file)
        if len(tbl) == 0:
            return -9999, -9999, -9999, -9999
    else:
        tbl = dat_file
    
    if GBT_band == "L":
        fch1 = 2251.46484375 # LBAND  --  based on the fch1 values from Table 6 of Lebofsky et al 2019
        num_course_channels = 512
    if GBT_band == "C":
        fch1 = 8438.96484375 # CBAND                           ""
        num_course_channels = 1664
    if GBT_band == "S":
        fch1 = 3151.46484375 # SBAND                           ""
        num_course_channels = 512
    if GBT_band == "X":
        fch1 = 11251.46484375 # XBAND                          ""
        num_course_channels = 1280
    foff = float(tbl["DELTAF"][0])*1e-6 
    
    nfpc=(1500.0/512.0)/abs(foff)
    
    return fch1, foff, nfpc, num_course_channels

def spike_channels(num_course_channels, nfpc):
    """makes a spike channels list given a list of channels"""
    spike_channels_list=[]
    for i in np.arange(num_course_channels): 
        spike_channel=(nfpc/2.0)+(nfpc*i)
        spike_channels_list.append(spike_channel)
    return spike_channels_list

def freqs_fine_channels(spike_channels_list, fch1, foff):
    freqs_fine_channels_list=[]
    for index, value in enumerate(spike_channels_list):
        freq_fine_channel=fch1+foff*value
        if freq_fine_channel>0:
            freq_fine_channel=round(freq_fine_channel, 6)
            freqs_fine_channels_list.append(freq_fine_channel)
        else:
            break
    return freqs_fine_channels_list

def clean_one_dat(datfile_curr, outpath, freqs_fine_channels_list, foff):
    """
    a function to encapsulate the code that identifies
    and removes the DC spikes
    """
    # set the outdir to the no_DC_spike directory
    split_file = datfile_curr.split("/")
    dat_name = split_file[len(split_file)-1]
    output = outpath + "/" + dat_name + "new.dat"
    
    file_contents = []
    # open file you wish to read
    with open(datfile_curr, 'r') as infile:
        for line in infile:
            file_contents.append(line)
    with open(output, 'w') as outfile:
        for index, row in enumerate(file_contents):
            if index<9:
                newrow=row.strip('\t')
                outfile.write(newrow)       
            else:
                newrow=row.split('\t')
                row=row.split('\t')
                freq=float(newrow[4])-(foff/2.0)
                startfreq=float(newrow[6])-(foff/2.0)
                endfreq=float(newrow[7])-(foff/2.0)
                freq=round(freq, 6)
                startfreq=round(startfreq,6)
                endfreq=round(endfreq,6)

                minfreq=(float(freq)-0.000001)
                maxfreq=(float(freq)+0.000001)
                minfreq=round(minfreq,6)
                maxfreq=round(maxfreq,6)
                freq=str(freq)
                minfreq=str(minfreq)
                maxfreq=str(maxfreq)
                if len((freq).split('.')[1])<6:
                    freq=format(float(freq), '.6f')
                row[3]=str(freq)
                row[4]=str(freq)
                row[6]=str(startfreq)
                row[7]=str(endfreq)
                string='\t'
                for index,value in enumerate(row[:-1]):
                    newvalue=value+string
                    row[index]=newvalue
                if len((minfreq).split('.')[1])<6:
                    minfreq=format(float(minfreq), '.6f')
                if len((maxfreq).split('.')[1])<6:
                    maxfreq=format(float(maxfreq), '.6f')
                bad_freq = float(freq) in freqs_fine_channels_list or float(minfreq) in freqs_fine_channels_list or float(maxfreq) in freqs_fine_channels_list
                if not bad_freq:
                    glue='  '
                    row=glue.join(row)
                    outfile.write(str(row))

def remove_DC_spike(dat_file, outdir, GBT_band):
    """
    The driver function which generates and saves 
    a .dat file without DC spikes
    
    Arguments
    ----------
    dat_file : str
        the .dat file which will be cleaned of DC spikes
    outdir : str
        the filepath to where the cleaned .dat files will be saved
    GBT_band : str
        the band at which the data was collected
        choose from {"L", "S", "C", "X"}
    num_course_channels : int
        the number of course channels in a frequency band. The 
        default is 512
    """
    fch1, foff, nfpc, num_course_channels = grab_parameters(dat_file, GBT_band)
    if fch1 == -9999:
        dat_name = os.path.basename(dat_file)
        empty_dat_command = "cp %s %s"%(dat_file, outdir + dat_name + "new.dat")
        os.system(empty_dat_command)
        return
    spike_channels_list = spike_channels(num_course_channels, nfpc)
    freqs_fine_channels_list = freqs_fine_channels(spike_channels_list,fch1, foff)
    clean_one_dat(dat_file, outdir, freqs_fine_channels_list, foff)
                
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Takes a set of .dat files and produces a new set of .dat files that have the DC spike removed. The files will be saved to a new directory that is created in the same directory as the .dat files, called <band>_band_no_DC_spike")
    parser.add_argument("band", help="the GBT band that the data was collected from. Either L, S, C, or X")
    parser.add_argument("-folder", "-f", help="directory .dat files are held in")
    parser.add_argument("-t", help="a .txt file to read the filepaths of the .dat files", action=None)
    parser.add_argument("-outdir", "-o", help="directory where the results are saved", default=".")
    args = parser.parse_args()

    # collect paths to .dat files
    if args.t == None:
        dat_files = glob.glob(args.folder+"/*.dat")
    else:
        dat_files = read_txt(args.t)

    # set the GBT band
    GBT_band = args.band
    
    # make a directory to store the .dats that have had the DC spike removed
    checkpath = args.outdir + "/%s_band_no_DC_spike"%args.band
    if os.path.isdir(checkpath):
        pass
    else:
        os.mkdir(checkpath)

    print("Removing DC spikes...")
    start = time.time()
    for i in trange(len(dat_files)):
        remove_DC_spike(dat_files[i], checkpath, GBT_band)
    end = time.time()

    print("All Done!")
    print("It took %s seconds to remove DC spikes from %s files"%(end - start, len(dat_files)))