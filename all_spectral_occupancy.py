import spectral_occupancy as so
import matplotlib.pyplot as plt
import glob
import numpy as np

outdir = "/datax/scratch/danielb/GBT_statistics/all_bldw/spectral_occupancy/"# "/home/danielb/GBT_statistics/spectral_occupancy_plots/"#

bands = ["L", "S", "C", "X"]

fig, axs = plt.subplots(4, 1, figsize=(10,10))
plt.subplots_adjust(hspace=0.6)
fontsize = 5
opacity = 0.25

for i in range(len(bands)):
    band_paths = glob.glob("/datax/scratch/danielb/GBT_statistics/all_bldw/%s_band_no_DC_spike/*_*dat"%bands[i])
    # bl tess targets
    # band_paths = glob.glob("/home/danielb/gbt_dats/traas_dats/%s_band_no_DC_spike/*_*dat"%bands[i])

    bin_edges, prob_hist, n_observations, ax = so.calculate_proportion(band_paths, bin_width=1, GBT_band=bands[i], notch_filter=True, outdir=outdir, exclude_zero_drift=True)
    axs[i].bar(bin_edges[:-1], prob_hist, width=1)
    if bands[i] == "L":
        axs[i].axvspan(1200, 1341, alpha=opacity, color='red', label="notch filter region")
        axs[i].axvspan(1164, 1215, alpha=opacity, color="tab:orange", label="GNSS")
        axs[i].axvspan(1350, 1390, alpha=opacity, color="tab:green", label="ATC")
        axs[i].axvspan(1525, 1535, alpha=opacity, color="tab:purple", label="MSS")
        axs[i].axvspan(1535, 1559, alpha=opacity, color="tab:brown", label="MSS")
        axs[i].axvspan(1559, 1626.5, alpha=opacity, color="tab:pink", label="GNSS")
        axs[i].axvspan(1675, 1695, alpha=opacity, color="tab:gray", label="GOES")
        axs[i].axvspan(1850, 1900, alpha=opacity, color="tab:olive", label="PCS")
        axs[i].legend(fontsize=fontsize, ncol=3)
    if bands[i] == "S":
        axs[i].axvspan(2300, 2360, alpha=opacity, color='red', label="notch filter region")
        axs[i].axvspan(1850, 2000, alpha=opacity, color="tab:orange", label="PCS")
        axs[i].axvspan(2025, 2035, alpha=opacity, color="tab:green", label="GOES")
        axs[i].axvspan(2100, 2120, alpha=opacity, color="tab:purple", label="DSN")
        axs[i].axvspan(2180, 2200, alpha=opacity, color="tab:brown", label="MSS")
        axs[i].axvspan(2200, 2290, alpha=opacity, color="tab:olive", label="EES")
        axs[i].legend(fontsize=fontsize, ncol=2)
    if bands[i] == "C":
        axs[i].axvspan(4000, 4200, alpha=opacity, color="tab:orange", label="FSS")
        axs[i].axvspan(4500, 4800, alpha=opacity, color="tab:orange")
        axs[i].legend(fontsize=fontsize)
    if bands[i] == "X":
        axs[i].axvspan(8025, 8400, alpha=opacity, color="tab:orange", label="ESS")
        axs[i].axvspan(10700, 11200, alpha=opacity, color="tab:green", label="FSS")
        axs[i].legend(fontsize=fontsize)
    axs[i].set_xlabel("Frequency [MHz]")
    axs[i].set_ylabel("Fraction with Hits")
    axs[i].set_title("%s Band Spectral Occupancy: n = %s observations"%(bands[i], n_observations))
    axs[i].set_ylim(0,1.05)
# fig.savefig("test.pdf", bbox_inches="tight", transparent=False)
fig.savefig(outdir + "all_bands_spectral_occupancy.pdf", bbox_inches="tight", transparent=False)
fig.savefig("/home/danielb/writeup_plots/all_bands_spectral_occupancy.pdf", bbox_inches="tight", transparent=False)
