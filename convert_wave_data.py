from __future__ import print_function

import argparse
from scipy.io import wavfile
from scipy.signal import resample
import matplotlib.pylab as pl
import numpy as np
import cPickle


def main_mono(args):
    # dfile = "explorers_MP3WRAP"
    # dfile = "drinksonus"
    dfile = args.file

    rate, data = wavfile.read("%s.wav" % dfile)

    print(data.shape)

    seqs = []
    si = 0 # sample index
    while si < data.shape[0]:
        # print("si = %d" % si)
        incr = int(np.random.normal(1000, 50))
        # check
        if incr < 100:
            print("too short %d" % incr)
            continue
        if si+incr >= data.shape[0]:
            print("adjusting final increment")
            incr = data.shape[0] - si
        seq = np.zeros((incr, 3))
        seq[:,0:2] = data[si:si+incr]
        seq[-1,2] = 1 # end of stroke
        seqs.append(seq.copy())
    
        si += incr

    # print("seqs", seqs)
    # print("len(seqs)", len(seqs))

    f = open("%s.cpkl" % dfile,"wb")
    cPickle.dump(seqs, f, protocol=2)
    f.close()

    # pl.plot(data[210000:220000])
    # pl.show()

def main_stereo(args):
    dfile = args.file

    rate, data = wavfile.read("%s.wav" % dfile)

    print(data.shape)

    print(data.dtype)
    print(np.min(data), np.max(data))
    
    seqs = []
    si = 0 # sample index
    while si < data.shape[0]:
        # print("si = %d" % si)
        # incr = int(np.random.normal(1000, 50))
        incr = 4410 # 1/10th second
        # check
        if incr < 100:
            print("too short %d" % incr)
            continue
        if si+incr >= data.shape[0]:
            # print("adjusting final increment")
            # incr = data.shape[0] - si
            # ditch it
            si += incr
            continue
        
        seq = np.zeros((incr, 2))
        seq[:,:] = data[si:si+incr]
        # seq[-1,2] = 1 # end of stroke
        seqs.append(seq.copy())
    
        si += incr

    # print("seqs", seqs)
    # print("len(seqs)", len(seqs))

    f = open("%s.cpkl" % dfile,"wb")
    cPickle.dump(seqs, f, protocol=2)
    f.close()
    
def main(args):
    if args.mode == "mono":
        main_mono(args)
    elif args.mode == "stereo":
        main_stereo(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", default="mono", help="Mode: mono, stereo")
    parser.add_argument("-f", "--file", default="drinksonus", help="wavfile to convert: [drinksonus]")

    args = parser.parse_args()

    main(args)
