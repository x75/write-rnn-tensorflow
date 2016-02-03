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

    print("rate", rate, "data.shape", data.shape)

    if args.eight:
        print("adjusting data offset")
        data = data.astype(np.int16)
        data -= 128
        print("data.min", np.min(data))
        
    seqs = []
    si = 0 # sample index
    incr = rate/10
    while si < data.shape[0]:
        # print("si = %d" % si)
        # incr = int(np.random.normal(1000, 50))
        incr = int(np.random.uniform(300, 800))
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

    print("data.shape", data.shape)
    if len(data.shape) < 2:
        # print("mono input?")
        data = np.vstack((data, data)).T

    print("data.shape", data.shape)
    print("data.dtype", data.dtype)
    print("data min max", np.min(data), np.max(data))

    if args.eight:
        print("adjusting data offset")
        data = data.astype(np.int16)
        data -= 128
        print("data.min", np.min(data))
        
    seqs = []
    si = 0 # sample index
    incr = rate/10 # 1/10th second
    print("incr", incr)
    while si < data.shape[0]:
        # print("si = %d" % si)
        # incr = int(np.random.normal(1000, 50))
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
        
        seq = np.zeros((incr, 2), dtype=np.int16)
        seq[:,:] = data[si:si+incr].astype(np.int16)
        # print("seq.dtype", seq.dtype, seq)
        # seq[-1,2] = 1 # end of stroke
        seqs.append(seq.copy())
    
        si += incr

    # print("seqs", seqs)
    # print("len(seqs)", len(seqs))

    f = open("%s.cpkl" % dfile,"wb")
    cPickle.dump(seqs, f, protocol=2)
    f.close()

# def main_8bit():
        
def main(args):
    if args.mode == "mono":
        main_mono(args)
    elif args.mode == "stereo":
        main_stereo(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", default="mono", help="Mode: mono, stereo, 8bit")
    parser.add_argument("-f", "--file", default="drinksonus", help="wavfile to convert: [drinksonus]")
    parser.add_argument("-8", "--eight", action="store_true", help="8bit data", dest="eight")

    args = parser.parse_args()
    print(args)
    main(args)
