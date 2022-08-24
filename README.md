# cycfold
This is a collection of scripts for folding GUPPI raw files (and, eventually, producing cyclic spectra from them).
The workhorse script is `fold.py`, which, when provided with a pulsar ephemeris in the form of a TEMPO Polyco, will fold the contents of a GUPPI raw file on the period of the pulsar.
It will output an `.npz` file containing the folded profiles and some metadata, which can then be post-processed and plotted using `view_folded.py`.

The code depends four Python packages: numpy, matplotlib, astropy, and [baseband](https://github.com/mhvk/baseband).
