from __future__ import print_function

import argparse
from scipy.io import wavfile
from scipy.signal import resample
import matplotlib.pylab as pl
import numpy as np
import cPickle

def gen_data(length=100):
    t = np.linspace(0, length, length, endpoint=False)
    s = np.zeros_like(t)
    Ncomp = 2
    for i in range(Ncomp):
        # s += np.sin(t * ((0.5*i) + np.random.uniform(-0.01, 0.01)))
        s += np.sin(t * (0.1*(i+1)))
    return (1./Ncomp) * s/np.max(np.abs(s))

def main(args):
    stereo = True # False
    s = gen_data(args.length).astype(np.float32)
    print(s.shape)
    if stereo:
        s_s = np.vstack((s, s)).T
    else:
        s_s = s
    s_s_int = (s_s * 32767).astype(np.int16)
    print(s_s.shape, s_s_int.shape)
    pl.subplot(211)
    pl.plot(s_s)
    pl.subplot(212)
    pl.plot(s_s_int)
    pl.show()
    wavfile.write("mso6s.wav", 44100, s_s_int)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--mode", default="mono", help="Mode: mono, stereo")
    # parser.add_argument("-f", "--file", default="drinksonus", help="wavfile to convert: [drinksonus]")
    parser.add_argument("-l", "--length", default=44100, type=int, help="Length of generated audio")

    args = parser.parse_args()

    main(args)
