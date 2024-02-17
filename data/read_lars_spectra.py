import numpy as np
import os, sys, pdb
import matplotlib.pyplot as plt

from astropy.io import fits
from matplotlib.animation import FuncAnimation

# define file paths and function to parse fits files
f1 = "lars_l12_20171019-082040_clv5434_mu10.ns.chvtt.fits"
f2 = "lars_l12_20171019-084048_clv5434_mu10.ns.chvtt.fits"

def read_spectrum(f):
    with fits.open(f) as hdul:
        # get the data
        flux1 = hdul[0].data[0,:,:].astype(float)
        nois1 = hdul[0].data[1,:,:].astype(float)
        wave1 = hdul[1].data

        # get the timestamps
        meta1 = hdul[2].data
        timestamps = [i[2] for i in meta1]

    return wave1, flux1, nois1, timestamps

# read in the files
wave1, flux1, nois1, timestamps1 = read_spectrum(f1)
wave2, flux2, nois2, timestamps2 = read_spectrum(f2)

# do a very, very simple normalization, scale the noise
for i in range(np.shape(flux1)[0]):
    nois1[i,:] /= np.max(flux1[i,:])
    flux1[i,:] /= np.max(flux1[i,:])

for i in range(np.shape(flux2)[0]):
    nois2[i,:] /= np.max(flux2[i,:])
    flux2[i,:] /= np.max(flux2[i,:])

# concatenate the time series
wavs_all = np.concatenate((wave1, wave2), axis=0)
flux_all = np.concatenate((flux1, flux2), axis=0)
nois_all = np.concatenate((nois1, nois2), axis=0)
timestamps_all = np.concatenate((timestamps1, timestamps2), axis=0)

# plot one slice
plt.plot(wave1[0,:], flux1[0,:], c="k")
plt.title(timestamps1[0])
plt.show()
plt.savefig('plot1.png')

# now make an animation
# initialize plot objects
fig, ax1 = plt.subplots()

# set the data limits
ax1.set_xlim(np.min(wavs_all) - 0.1, np.max(wavs_all) + 0.1)
ax1.set_ylim(0.1, 1.1)

# initialize the line
l1, = ax1.plot([], [], c="k")

# define animator function
def animator(i):
    ax1.set_title(timestamps_all[i])
    l1.set_data(wavs_all[i,:], flux_all[i,:])
    return l1,

ani = FuncAnimation(fig, animator, frames=len(timestamps_all), blit=True)
plt.show()
