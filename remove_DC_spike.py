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
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time

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

def grab_parameters(dat_file, GBT_band, outdir):
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
    outdir : str
        the filepath to where the csv containing the RA, 
        DEC, MJD data will be saved
        
    Returns : fch1, foff, nfpc
    which will be used internally within remove_DC_spike
    """
    if type(dat_file) != pandas.core.frame.DataFrame:
        tbl = find.read_dat(dat_file)
    else:
        tbl = dat_file

    dat_name = os.path.basename(dat_file)
    filepath = outdir + "/" + dat_name + "new.dat"

    if len(tbl) == 0: # return -9999 then the RA, DEC, MJD
        with open(dat_file, "r") as f:
            hits = f.readlines()
        MJD = hits[4].strip().split('\t')[0].split(':')[-1].strip()
        RA = hits[4].strip().split('\t')[1].split(':')[-1].strip()
        DEC = hits[4].strip().split('\t')[2].split(':')[-1].strip()
        alt, az = get_AltAz(RA, DEC, MJD)
        # extract hour of day from MJD and record to csv
        hour = Time(MJD, format="mjd").datetime.hour - 5 # adjust time zone from UTC to eastern (UTC-5)
        with open(outdir + "/locations.csv", "a") as f:
            f.write(str(RA) + "," + str(DEC) + "," + str(MJD) + "," + str(alt) + "," + str(az) + "," + str(hour) + "," + filepath + "\n")
        return -9999, -9999, -9999, -9999
    else:
        RA = tbl["RA"].values[0]
        DEC = tbl["DEC"].values[0]
        MJD = tbl["MJD"].values[0]
        alt, az = get_AltAz(RA, DEC, MJD)
        # extract hour of day from MJD and record to csv
        hour = Time(MJD, format="mjd").datetime.hour
        with open(outdir + "/locations.csv", "a") as f:
            f.write(str(RA) + "," + str(DEC) + "," + str(MJD) + "," + str(alt) + "," + str(az) + "," + str(hour) + "," + filepath + "\n")
    
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
    dat_name = os.path.basename(datfile_curr)
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
    csv_header = "RA,DEC,MJD,ALT,AZ,hour (UTC - 5),filepath\n"
    if not os.path.exists(outdir + "/locations.csv"):
        with open(outdir + "/locations.csv", "w") as f:
            f.write(csv_header)
    fch1, foff, nfpc, num_course_channels = grab_parameters(dat_file, GBT_band, outdir)
    if fch1 == -9999:
        dat_name = os.path.basename(dat_file)
        empty_dat_command = "cp %s %s"%(dat_file, outdir + "/" + dat_name + "new.dat")
        os.system(empty_dat_command)
        return
    spike_channels_list = spike_channels(num_course_channels, nfpc)
    freqs_fine_channels_list = freqs_fine_channels(spike_channels_list,fch1, foff)
    clean_one_dat(dat_file, outdir, freqs_fine_channels_list, foff)
                
def get_AltAz(RA, DEC, MJD):
    targets = SkyCoord(RA, DEC, unit=(u.hourangle, u.deg), frame="icrs")          
    times = Time(np.array(MJD, dtype=float), format="mjd") 
    gbt = EarthLocation(lat=38.4*u.deg, lon=-79.8*u.deg, height=808*u.m)
    gbt_altaz_transformer = AltAz(obstime=times, location=gbt)
    gbt_target_altaz = targets.transform_to(gbt_altaz_transformer)
    altitude = gbt_target_altaz.alt.deg
    azimuth = gbt_target_altaz.az.deg

    return altitude, azimuth

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Takes a set of .dat files and produces a new set of .dat files that have the DC spike removed. The files will be saved to a new directory that is created in the same directory as the .dat files, called <band>_band_no_DC_spike")
    parser.add_argument("band", help="the GBT band that the data was collected from. Either L, S, C, or X")
    parser.add_argument("-folder", "-f", help="directory .dat files are held in")
    parser.add_argument("-t", help="a .txt file to read the filepaths of the .dat files", action=None)
    parser.add_argument("-outdir", "-o", help="directory where the results are saved", default=os.getcwd())
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

    # check if any of the files have already been cleaned
    to_clean = []
    for i in range(len(dat_files)):
        one_dat = os.path.basename(dat_files[i]) + "new.dat"
        one_path = checkpath + "/" + one_dat
        if not os.path.exists(one_path):
            to_clean.append(dat_files[i])

    print("Removing DC spikes...")
    start = time.time()
    for i in trange(len(to_clean)):
        remove_DC_spike(to_clean[i], checkpath, GBT_band)
    end = time.time()

    print("All Done!")
    print("It took %s seconds to remove DC spikes from %s files"%(end - start, len(to_clean)))