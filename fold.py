import numpy as np
import baseband
import astropy.units as u
import time

from polyco import Polyco
from cli import ProgressBar

def fold(rawfile, polyco, nchan, nbin, nblock=65536, quiet=False, pb_steps=2):
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
    nchan_per = nchan//nchan_raw  # number of channels to make for each raw channel
    buffer = np.zeros((4, nchan_raw, nchan_per, nbin)) # polarizations, raw channels, sub-channels, bins
    if (not quiet):
        progress_bar = ProgressBar(width=52, steps=pb_steps)
    while True:
        # Calculate the block size and number of segments
        pos = rawfile.tell()
        remaining = rawfile.shape[0] - pos
        if remaining == 0:
            break
        block_size = min(nblock, nchan_per*(remaining//nchan_per))
        nseg = block_size//nchan_per # number of segments of length nchan_per in the block

        # Calculate the phase bin corresponding to each segment
        idx_seg = np.arange(pos + nchan_per//2, pos + nchan_per*nseg, nchan_per)
        t_seg = rawfile.start_time + idx_seg/rawfile.sample_rate
        dphs_seg = polyco.dphase(t_seg.to_value('mjd', 'long'))
        iphs_seg = np.floor(nbin*(dphs_seg % 1)).astype(np.int64)
        
        # Read in a block of samples of the previously calculated size
        block = rawfile.read(block_size)
        # block shape at this point is (nblock, npol=2, nchan_raw)
        # Divide the block into nseg segments of length nchan_per
        block = block.transpose(1, 2, 0).reshape(2, nchan_raw, -1, nchan_per)
        # new block shape is (npol=2, nchan_raw, nseg, nchan_per)

        # Make a filterbank by taking an FFT of length nchan_per (along the last axis)
        X = np.fft.fft(block[0])
        Y = np.fft.fft(block[1])
        
        # Detect and write coherence data to buffer
        XX = np.abs(X)**2
        YY = np.abs(Y)**2
        CC = X*np.conj(Y)
        CR = np.real(CC)
        CI = np.imag(CC)
        for i in range(nseg):
            buffer[0,:,:,iphs_seg[i]] += XX[:,i]
            buffer[1,:,:,iphs_seg[i]] += YY[:,i]
            buffer[2,:,:,iphs_seg[i]] += CR[:,i]
            buffer[3,:,:,iphs_seg[i]] += CI[:,i]

        if not quiet:
            progress = rawfile.tell()/rawfile.shape[0]
            progress_sec = (rawfile.tell()/rawfile.sample_rate).to(u.s)
            progress_bar.update(progress, f"Processed {progress_sec:.1f} of {obslen:.1f} ({progress:.1%})")
    if not quiet:
        print()
    buffer = np.fft.fftshift(buffer, axes=2)
    buffer = buffer.reshape(4, nchan, nbin)
    nseg_total = rawfile.shape[0]//nchan_per
    buffer /= nseg_total
    return buffer

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--quiet', action='store_true', help="don't display progress messages")
    parser.add_argument('-P', dest='predictor', type=str, help="phase predictor (polyco file) to be used for folding")
    parser.add_argument('-F', dest='nchan', type=int, help="number of frequency channels to create")
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
        print(f"Folding data from {args.rawfile} into {args.nchan} channels and {args.nbin} phase bins...")

    rawfile = baseband.open(args.rawfile)
    obsfreq = rawfile.header0['OBSFREQ']
    obsbw = rawfile.header0['OBSBW']
    chan_bw = rawfile.header0['CHAN_BW']
    chan_dm = rawfile.header0['CHAN_DM']
    mjd_start = rawfile.start_time.to_value('mjd', 'long')
    mjd_stop = rawfile.stop_time.to_value('mjd', 'long')
    ref_f0 = polyco.f0((mjd_start + mjd_stop)/2)
    tbin = float(1/(args.nbin*ref_f0))
    freq_edges = obsfreq - obsbw/2 + np.arange(args.nchan + 1) * obsbw / args.nchan
    freq_centers = obsfreq - obsbw/2 + (np.arange(args.nchan) + 1/2) * obsbw / args.nchan
    phase_edges = np.linspace(0, 1, args.nbin + 1)

    start_time = time.perf_counter()
    buffer = fold(rawfile, polyco, args.nchan, args.nbin, args.nblock, args.quiet, pb_steps=8)
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
