import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import baseband
from baseband import guppi
import astropy.units as u

def fft_roll(a, shift):
    """
    Roll array by a given (possibly fractional) amount, in bins.
    Works by multiplying the FFT of the input array by exp(-2j*pi*shift*f)
    and Fourier transforming back. The sign convention matches that of 
    numpy.roll() -- positive shift is toward the end of the array.
    This is the reverse of the convention used by pypulse.utils.fftshift().
    If the array has more than one axis, the last axis is shifted.
    """
    n = a.shape[-1]
    phase = -2j*np.pi*shift*np.fft.rfftfreq(n)
    return np.fft.irfft(np.fft.rfft(a)*np.exp(phase), n)

def dedisperse(arr, freq, tbin, DM, ref_freq):
    """
    Dedisperse data in an array according to a corresponding set of frequencies.
    
    Parameters:
    -----------
    arr:  Array containing the data to be dedispersed, with phase axis last.
    freq: Array of frequencies corresponding to the frequency axis of arr.
    tbin: Time, in seconds, corresponding to a single phase bin in arr.
    DM:   The dispersion measure at which to dedisperse.
    ref_freq: The reference frequency for the dedispersion.
    """
    # TEMPO/Tempo2/PRESTO conventional value (cf. Kulkarni 2020, arXiv:2007.02886)
    K = 1/2.41e-4 # s MHz**2 cm**3 / pc
    time_delays = K*DM*(freq**-2 - ref_freq**-2)
    bin_delays = time_delays/tbin
    return fft_roll(arr, -bin_delays[..., np.newaxis])

def plot_freq_phase(ax, arr, freq, phase, label=""):
    """
    Plot a set of profiles vs. frequency and phase.
    
    Parameters:
    ax:    Axis for plotting.
    arr:   Array of profiles, with frequency axis first and phase axis second.
    freq:  Edges of frequency channels, in MHz.
    phase: Edges of phase bins, in cycles.
    label: Label for color bar.
    """
    pc = ax.pcolormesh(phase, freq, arr)
    cb = plt.colorbar(pc)
    ax.set_xlabel("Phase (cycles)")
    ax.set_ylabel("Frequency (MHz)")
    cb.set_label(label)

def offpulse_window(arr, size=None, frac=1/8):
    '''
    Find the off-pulse window of a given profile or set of profiles, defined as the
    segment of pulse phase of a given length minimizing the integral of the pulse
    profile. The length of the segment can be given as a number of pulse phase bins
    (`size`) or as a fraction of the period (`frac`). If `size` is given explicitly,
    `frac` is ignored. If a multidimensional array is given as input, the last axis
    is treated as the phase axis.
    '''
    if size is None:
        size = int(frac*arr.shape[-1])
    bins = np.arange(arr.shape[-1])
    lower = np.argmin(rolling_mean(arr, size), axis=-1)
    upper = lower + size
    try:
        lower = lower[..., np.newaxis]
        upper = upper[..., np.newaxis]
    except (TypeError, IndexError):
        pass
    return np.logical_and(lower <= bins, bins < upper)

def rolling_mean(arr, size):
    '''
    Calculate the mean of values in `arr` in a sliding window of length `size`,
    wrapping around at the end of the array. If `arr` is more than one-dimensional,
    the rolling mean will be computed over the last dimension only.
    '''
    n = arr.shape[-1]
    filtr = np.zeros(n)
    filtr[-size:] = 1
    return np.fft.irfft(np.fft.rfft(arr)*np.fft.rfft(filtr))/size

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('npzfile', type=str, help=".npz file to view")
    parser.add_argument('-d', dest='DM', type=float, default=None, help="dedisperse with the given DM")
    args = parser.parse_args()

    npz = np.load(args.npzfile)
    folded_profiles = npz['folded_profiles']
    AA, BB, CR, CI = folded_profiles
    freq_edges = npz['freq_edges']
    freq_centers = npz['freq_centers']
    phase_edges = npz['phase_edges']
    chan_dm = npz['chan_dm']
    tbin = npz['tbin']
    obsfreq = npz['obsfreq']
    print(chan_dm)

    if args.DM is not None:
        DM = args.DM
    else:
        DM = chan_dm

    I = dedisperse(AA + BB, freq_centers, tbin, DM, obsfreq)
    opw = offpulse_window(I.mean(axis=0))
    opmean = np.mean(I[:, opw], axis=-1)
    I -= opmean[:, np.newaxis]
    
    fig, ax = plt.subplots()
    plot_freq_phase(ax, I, freq_edges, phase_edges, "I (excess digitizer counts)")
    plt.show()
