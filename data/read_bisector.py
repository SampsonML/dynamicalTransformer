# ---
# read the bisector data
# ---

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import cmasher as cmr

# ----------------------------------------------- #
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.major.size"] = 7
plt.rcParams["xtick.minor.size"] = 4.5
plt.rcParams["ytick.major.size"] = 7
plt.rcParams["ytick.minor.size"] = 4.5
plt.rcParams["xtick.major.width"] = 2
plt.rcParams["xtick.minor.width"] = 1.5
plt.rcParams["ytick.major.width"] = 2
plt.rcParams["ytick.minor.width"] = 1.5
plt.rcParams["axes.linewidth"] = 2
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams.update({"text.usetex": True})
# ----------------------------------------------- #


def read_bisector(filename, masknans=False):
    with fits.open(filename) as hdulist:
        # λs in 1st extension, bis value in 2nd
        wav = hdulist[0].data
        bis = hdulist[1].data

    # mask NaNs
    if masknans:
        mask = ~np.isnan(wav)
        wav = wav[mask]
        bis = bis[mask]

    return wav, bis


filename = "input_data/FeI_5434/bisectors/lars_l20_20160819-144612_clv5434_mu10.ns.chvtt.FeI5434.5_norm.bisect.fits"
filename2 = "input_data/FeI_5434/bisectors/lars_l20_20160819-144612_clv5434_mu10.ns.chvtt.FeI5434.5_norm.bisect.fits"
filename3 = "input_data/FeI_5434/bisectors/lars_l20_20160820-084056_clv5434_mu09_s.ns.chvtt.FeI5434.5_norm.bisect.fits"
filename4 = "input_data/FeI_5434/bisectors/lars_l20_20171019-171245_clv5434_mu095_s.ns.chvtt.FeI5434.5_norm.bisect.fits"

wav, bis = read_bisector(filename2, masknans=False)
bis = bis / 60  # convert to minutes

plot_nums = 40 #len(bis)
color_nums = 10
cmap = plt.get_cmap("cmr.ember", plot_nums)
cols = [cmap(1 * i / plot_nums) for i in range(plot_nums)]

plt.figure(figsize=(9, 9), dpi=150)
for idx in range(0, plot_nums):
    plt.plot(np.log(wav[0][idx] * 0.1), wav[1][idx], c=cols[idx], alpha=0.75, zorder=0)
x_limits = plt.xlim()
x = np.linspace(0.5 * x_limits[0], 2 * x_limits[1], 1000)
plt.ylim([0.15, 1])
plt.xlim([x_limits[0], x_limits[1]])
plt.fill_between(x, 0.8, 1, color="gray", alpha=0.95, zorder=1)
sm = ScalarMappable(cmap=cmap)
sm.set_array([bis[0:plot_nums]])
plt.colorbar(sm, label="Time (minutes)")
plt.xlabel("Wavelength log[nm]")
plt.ylabel("Normalised Intensity")
# plt.xscale('log')

from matplotlib.ticker import StrMethodFormatter

# plt.gca().yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))  # No decimal places
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.6f}')) # 2 decimal places
# plt.ticklabel_format(useOffset=False, style='plain')
plt.title("FeI 5434 Line Bisector")
plt.savefig("test_bisector.png")
