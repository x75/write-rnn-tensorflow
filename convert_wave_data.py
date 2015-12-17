from __future__ import print_function

from scipy.io import wavfile
import matplotlib.pylab as pl
import numpy as np
import cPickle

# dfile = "explorers_MP3WRAP"
dfile = "drinksonus"

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
