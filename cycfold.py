import numpy as np
import scipy.signal
import baseband
import astropy.units as u
import numba
import time

from polyco import Polyco
from cli import ProgressBar

def fold(rawfile, polyco, ncyc, nbin, nblock=65536, quiet=False, pb_steps=2):
    """
    Fold data from a GUPPI raw file to produce profiles with a specified number of frequency channels and phase bins.
    
    Parameters
    ----------
    rawfile:  The file object returned by baseband.open().
    polyco:   Polyco object for phase calculation.
    nchan:    Total number of frequency channels to produce.
    nbin:     Number of phase bins to use.
    nblock:   Number of samples to read in a time (per polarization & voltage channel).
              Changing this can have a significant effect on performance.
    quiet:    If True, don't print progress messages.
    pb_steps: Number of progress bar steps per text column. Can be 1, 2, 4, or 8.
              Depending on font support, values above 2 might not display as intended.
    """
    obslen = (rawfile.shape[0]/rawfile.sample_rate).to(u.s) # Length of observation in seconds
    nchan_pfb = rawfile.shape[2]  # number of voltage channels in the raw file
    buffer = np.zeros((4, nchan_pfb, 2*ncyc, nbin), dtype=np.complex128) # polarizations, filterbank channels, cyclic channels, bins
    if (not quiet):
        progress_bar = ProgressBar(width=52, steps=pb_steps)
    while True:
        # Calculate the block size and number of segments
        pos = rawfile.tell()
        remaining = rawfile.shape[0] - pos
        if remaining < 2*ncyc:
            break
        block_size = min(nblock, ncyc*(remaining//ncyc))
        nseg = block_size//ncyc - 1 # number of segments of length 2*ncyc (overlapping by ncyc) in the block

        # Calculate the phase bin corresponding to each segment
        idx_seg = np.arange(pos + ncyc, pos + ncyc*(nseg + 1), ncyc) # index of the center sample of each segment
        t_seg = rawfile.start_time + idx_seg/rawfile.sample_rate # convert index to time
        dphs_seg = polyco.dphase(t_seg.to_value('mjd', 'long')) # phase offset at center of each segment
        iphs_seg = np.floor(nbin*(dphs_seg % 1)).astype(np.int64) # index of phase bin that each semgment falls into
        
        # Read in a block of samples of the previously calculated size
        block = rawfile.read(block_size)
        # block shape at this point is ((nseg + 1)*ncyc, npol=2, nchan_pfb)
        npol = 2
        # Divide the block into nseg segments of length 2*ncyc, overlapping by ncyc
        s = block.dtype.itemsize
        block = np.lib.stride_tricks.as_strided(
            block,
            shape=(nseg, 2*ncyc, npol, nchan_pfb),
            strides=(ncyc*npol*nchan_pfb*s, npol*nchan_pfb*s, nchan_pfb*s, s),
        )
        # new block shape is (nseg, 2*ncyc, npol=2, nchan_pfb)
        block = block.transpose(2, 3, 0, 1)
        # new block shape is (npol=2, nchan_pfb, nseg, 2*ncyc)

        # Calculate the correlation function
        for i in range(nchan_pfb):
            for j in range(nseg):
                x = block[0, i, j]
                y = block[0, i, j]
                Rxx = scipy.signal.correlate(x[ncyc:], x, mode='valid')
                Ryy = scipy.signal.correlate(y[ncyc:], y, mode='valid')
                Rxy = scipy.signal.correlate(x[ncyc:], y, mode='valid') # TODO: fix this
                Cxx = np.fft.fftshift(np.fft.hfft(Rxx))
                Cyy = np.fft.fftshift(np.fft.hfft(Ryy))
                Cxy = np.fft.fftshift(np.fft.hfft(Rxy))
                buffer[0, i, :, iphs_seg[j]] += np.abs(Cxx)**2
                buffer[1, i, :, iphs_seg[j]] += np.abs(Cyy)**2
                buffer[2, i, :, iphs_seg[j]] += Cxy.real
                buffer[3, i, :, iphs_seg[j]] += Cxy.imag

        if not quiet:
            progress = rawfile.tell()/rawfile.shape[0]
            progress_sec = (rawfile.tell()/rawfile.sample_rate).to(u.s)
            progress_bar.update(progress, f"Processed {progress_sec:.1f} of {obslen:.1f} ({progress:.1%})")
    if not quiet:
        print()
    nseg_total = rawfile.shape[0]//ncyc - 1
    buffer /= nseg_total
    return buffer

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--quiet', action='store_true', help="don't display progress messages")
    parser.add_argument('-P', dest='predictor', type=str, help="phase predictor (polyco file) to be used for folding")
    parser.add_argument('-c', dest='ncyc', type=int, help="number of cyclic channels to create will be twice this")
    parser.add_argument('-b', dest='nbin', type=int, help="number of phase bins in folded profile")
    parser.add_argument('-B', dest='nblock', type=int, default=65536, help="number of samples to read at a time (default: 65536).")
    parser.add_argument('-O', dest='outfile', type=str, help="where to save output (.npz)")
    parser.add_argument('rawfile', type=str, help="GUPPI raw file to fold")
    args = parser.parse_args()
    
    with open(args.predictor) as f:
        lines = f.readlines()
        polycos = Polyco.parse(lines)
    polyco = polycos[0]
    
    if not args.quiet:
        print(f"Folding data from {args.rawfile}.")

    rawfile = baseband.open(args.rawfile)
    obsfreq = rawfile.header0['OBSFREQ']
    obsbw = rawfile.header0['OBSBW']
    chan_bw = rawfile.header0['CHAN_BW']
    chan_dm = rawfile.header0['CHAN_DM']
    mjd_start = rawfile.start_time.to_value('mjd', 'long')
    mjd_stop = rawfile.stop_time.to_value('mjd', 'long')
    ref_f0 = polyco.f0((mjd_start + mjd_stop)/2)
    tbin = float(1/(args.nbin*ref_f0))
    nchan_pfb = rawfile.shape[2]
    nchan = 2*args.ncyc*nchan_pfb
    freq_edges = obsfreq - obsbw/2 + np.arange(nchan + 1) * obsbw / nchan
    freq_centers = obsfreq - obsbw/2 + (np.arange(nchan) + 1/2) * obsbw / nchan
    phase_edges = np.linspace(0, 1, args.nbin + 1)

    if not args.quiet:
        print(f"Creating cyclic spectrum with {nchan} channels and {args.nbin} phase bins...")

    start_time = time.perf_counter()
    buffer = fold(rawfile, polyco, args.ncyc, args.nbin, args.nblock, args.quiet, pb_steps=8)
    end_time = time.perf_counter()
    print(f"Folding completed in {end_time - start_time:.1f} s.")
    
    if not args.quiet:
        print(f"Saving output to {args.outfile}...")
    np.savez(
        args.outfile,
        folded_profiles=buffer,
        freq_edges=freq_edges,
        freq_centers=freq_centers,
        phase_edges=phase_edges,
        obsfreq=obsfreq,
        obsbw=obsbw,
        chan_bw=chan_bw,
        chan_dm=chan_dm,
        mjd_start=mjd_start,
        mjd_stop=mjd_stop,
        ref_f0=ref_f0,
        tbin=tbin,
    )

    print("Done.")
