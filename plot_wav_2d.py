import sys
import pylab as pl
import numpy as np
from scipy.io import wavfile

def pl2d(d):
    num_strokes = 10
    chunk = 20 
    for i in range(num_strokes):
        sl = slice(i*chunk, (i+1)*chunk)
        pl.subplot(num_strokes/2, num_strokes/2, i+1)
        pl.plot(d[sl,0], d[sl,1], "k-,", lw=0.2)
        pl.gca().set_aspect(1)
    pl.show()

def pl1d(d):
    span = min(d.shape[0], 44100)
    pl.plot(d[:span])
    pl.show()

if __name__ == "__main__":
    fname = sys.argv[1]
    print fname
    r, d = wavfile.read(fname)

    # pl2d(d)
    pl1d(d)


