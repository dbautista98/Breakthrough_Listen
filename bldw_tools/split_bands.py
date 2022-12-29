import bldw
from bands import BAND_NAMES
from astropy.time import Time
from tqdm import trange
from multiprocessing import Pool
from turbo_seti import find_event as find

# txt_path = "/datax/scratch/danielb/GBT_statistics/all_bldw/dump.txt"
outdir = "/datax/scratch/danielb/GBT_statistics/all_bldw/spacetrack/"
txt_path = outdir + "dump.txt"

with open(txt_path, "r") as f:
    paths = f.readlines()

for i in range(len(paths)):
    paths[i] = paths[i].replace("\n", "")

c = bldw.Connection()

for i in trange(len(paths)):
    file_dat = open(paths[i].strip())
    hits = file_dat.readlines()
    MJD = hits[4].strip().split('\t')[0].split(':')[-1].strip()

    unix_time = Time(MJD, format="mjd").unix
    try:
        obs = c.guess_observation_by_timestamp(unix_time)
        receiver_id = obs.receiver_id
        receiver = c.fetch_receiver(receiver_id)
        band = BAND_NAMES[receiver.name]
        with open(outdir + "/%s_band_dats.txt"%band, "a") as f:
            f.write(paths[i] + "\n")
    except:
        with open(outdir + "failed_time_ID.txt", "a") as f:
            f.write(paths[i] + "\n")
