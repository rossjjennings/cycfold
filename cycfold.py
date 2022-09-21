import numpy as np
import baseband
import astropy.units as u
import numba
import time

from polyco import Polyco
from cli import ProgressBar

@numba.jit(nopython=True)
def half_corr(x, y):
    """
    Calculate the "half correlation" between two arrays of the same length.
    This is an estimate of the cross-correlation computed only at lags for which
    the arrays overlap by at least half their length (rounded down). When the length,
    n, is even, there are n+1 lags with this property; when it is odd, there are n.
    The effective number of samples averaged for each lag is equal to the minimum
    overlap; for every other lag, this means that two boundary samples have to be
    weighted by 1/2.
    """
    if y.size != x.size:
        raise ValueError("sizes of x and y do not match")
    n = x.size
    out_dtype = (x[0:1]*y[0:1]).dtype
    if n % 2 == 0:
        corr = np.zeros(n + 1, dtype=out_dtype)

        #narrow
        for i in range(n//2 + 1):
            for j in range(n//2):
                corr[2*i] += x[i + j]*np.conj(y[n//2 - i + j])

        #wide
        for i in range(n//2):
            corr[2*i + 1] += x[i]*np.conj(y[n//2 - 1 - i])/2 # half of j=0 term
            for j in range(1, n//2):
                corr[2*i + 1] += x[i + j]*np.conj(y[n//2 - 1 - i + j])
            corr[2*i + 1] += x[n//2 + i]*np.conj(y[n - 1 - i])/2 # half of j=n//2 term

        corr /= n//2
    else:
        corr = np.zeros(n, dtype=out_dtype)

        #wide
        for i in range((n+1)//2):
            corr[2*i] += x[i]*np.conj(y[(n-1)//2 - i])/2 # half of j=0 term
            for j in range(1, (n-1)//2):
                corr[2*i] += x[i + j]*np.conj(y[(n-1)//2 - i + j])
            corr[2*i] += x[(n-1)//2 + i]*np.conj(y[n - 1 - i])/2 # half of j=(n-1)//2 term

        #narrow
        for i in range((n-1)//2):
            for j in range((n-1)//2):
                corr[2*i + 1] += x[1 + i + j]*np.conj(y[(n-1)//2 - i + j])

        corr /= (n-1)//2
    return corr

@numba.jit(nopython=True)
def corr_from_block(block, iphs_seg, nchan_raw, ncyc, nbin):
    nseg = iphs_seg.shape[0]
    corr = np.zeros((4, nchan_raw, 2*ncyc + 1, nbin), dtype=np.complex128)
    for i in range(nchan_raw):
        for j in range(nseg):
            x = block[j, :, 0, i]
            y = block[j, :, 1, i]
            xx = np.fft.fftshift(np.fft.fft(half_corr(x, x)))
            yy = np.fft.fftshift(np.fft.fft(half_corr(y, y)))
            cc = np.fft.fftshift(np.fft.fft(half_corr(x, y)))
            cr = cc.real
            ci = cc.imag
            corr[0, i, :, iphs_seg[i]] += xx
            corr[1, i, :, iphs_seg[i]] += yy
            corr[2, i, :, iphs_seg[i]] += cr
            corr[3, i, :, iphs_seg[i]] += ci
    return corr

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
    nchan_raw = rawfile.shape[2]  # number of voltage channels in the raw file
    buffer = np.zeros((3, nchan_raw, 2*ncyc + 1, nbin), dtype=np.complex128) # polarizations, filterbank channels, cyclic channels, bins
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
        # block shape at this point is ((nseg + 1)*ncyc, npol=2, nchan_raw)
        npol = 2
        # Divide the block into nseg segments of length 2*ncyc, overlapping by ncyc
        s = block.dtype.itemsize
        block = np.lib.stride_tricks.as_strided(
            block,
            shape=(nseg, 2*ncyc, npol, nchan_raw),
            strides=(ncyc*npol*nchan_raw*s, npol*nchan_raw*s, nchan_raw*s, s),
        )
        # new block shape is (npol=2, nchan_raw, nseg, nchan_per)

        # Make a filterbank by taking an FFT of length nchan_per (along the last axis)
        output = corr_from_block(block, iphs_seg, nchan_raw, ncyc, nbin)
        buffer += output

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
    parser.add_argument('-c', dest='ncyc', type=int, help="number of cyclic channels to create will be twice this, plus one")
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
        print(f"Folding data from {args.rawfile} into cyclic spectrum with {2*args.ncyc + 1} channels and {args.nbin} phase bins...")

    rawfile = baseband.open(args.rawfile)
    obsfreq = rawfile.header0['OBSFREQ']
    obsbw = rawfile.header0['OBSBW']
    chan_bw = rawfile.header0['CHAN_BW']
    chan_dm = rawfile.header0['CHAN_DM']
    mjd_start = rawfile.start_time.to_value('mjd', 'long')
    mjd_stop = rawfile.stop_time.to_value('mjd', 'long')
    ref_f0 = polyco.f0((mjd_start + mjd_stop)/2)
    tbin = float(1/(args.nbin*ref_f0))
    freq_edges = obsfreq - obsbw/2 + np.arange(2*args.ncyc + 2) * obsbw / (2*args.ncyc + 1)
    freq_centers = obsfreq - obsbw/2 + (np.arange(2*args.ncyc + 1) + 1/2) * obsbw / (2*args.ncyc + 1)
    phase_edges = np.linspace(0, 1, args.nbin + 1)

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
