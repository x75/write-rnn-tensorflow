import sys, argparse
import pylab as pl
import numpy as np
from scipy.io import wavfile

def pl2d(args):
    num_strokes = 10
    chunk = 20 
    r, d = wavfile.read(args.wavfile)
    for i in range(num_strokes):
        sl = slice(i*chunk, (i+1)*chunk)
        pl.subplot(num_strokes/2, num_strokes/2, i+1)
        pl.plot(d[sl,0], d[sl,1], "k-,", lw=0.2)
        pl.gca().set_aspect(1)
    pl.show()

def pl1d(args):
    r, d = wavfile.read(args.wavfile)
    start = max(0, args.start)
    span = min(d.shape[0], args.length)
    pl.title(args.wavfile)
    pl.plot(d[start:start+span])
    pl.xlabel("Sample #")
    pl.ylabel("Amplitude")
    if args.saveplot:
        pl.gcf().set_size_inches(8, 4)
        pl.gcf().savefig("plot_wav_2d_%s_%d_%d.pdf" % (args.wavfile.replace("/", "_").replace(".", "_"), start, start+span), dpi=300, bbox_inches="tight")
    pl.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode",    type=str, default="mono")
    parser.add_argument("-w", "--wavfile", type=str, default=None)
    parser.add_argument("-l", "--length",  type=int, default=1000)
    parser.add_argument("-s", "--start",   type=int, default=0)
    parser.add_argument("-sp", "--saveplot", action="store_true")

    args = parser.parse_args()
    # fname = sys.argv[1]
    # print fname
    # r, d = wavfile.read(args.wavfile)

    if args.mode == "mono":
        pl1d(args)
    else:
        pl2d(args)


