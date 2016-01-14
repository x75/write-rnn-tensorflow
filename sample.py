import numpy as np
import tensorflow as tf

import time
import os
import cPickle
import argparse

from utils import *
from model import Model
from modela import Model2Df
import random


import svgwrite
from IPython.display import SVG, display

# main code (not in a main function since I want to run this script in IPython as well).

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default='sample',
                   help='filename of .svg file to output, without .svg')
parser.add_argument('--sample_length', type=int, default=800,
                   help='number of strokes to sample')
parser.add_argument('--model', type=str, default=None,
                   help='number of strokes to sample')
parser.add_argument('--scale_factor', type=int, default=10,
                   help='factor to scale down by for svg output.  smaller means bigger output')
sample_args = parser.parse_args()

with open(os.path.join('save', 'config.pkl')) as f:
    saved_args = cPickle.load(f)

# model = Model(saved_args, True)
model = Model2Df(saved_args, True)
sess = tf.InteractiveSession()
saver = tf.train.Saver(tf.all_variables())

if sample_args.model != None:
    print "loading model: %s" % (sample_args.model)
    saver.restore(sess, sample_args.model)
else:  # use latest saved checkpoint
    ckpt = tf.train.get_checkpoint_state('save')
    print "loading model: ",ckpt.model_checkpoint_path
    saver.restore(sess, ckpt.model_checkpoint_path)

def sample_stroke():
  [strokes, params] = model.sample(sess, sample_args.sample_length)
  draw_strokes(strokes, factor=sample_args.scale_factor, svg_filename = sample_args.filename+'.normal.svg')
  draw_strokes_random_color(strokes, factor=sample_args.scale_factor, svg_filename = sample_args.filename+'.color.svg')
  draw_strokes_random_color(strokes, factor=sample_args.scale_factor, per_stroke_mode = False, svg_filename = sample_args.filename+'.multi_color.svg')
  draw_strokes_eos_weighted(strokes, params, factor=sample_args.scale_factor, svg_filename = sample_args.filename+'.eos_pdf.svg')
  draw_strokes_pdf(strokes, params, factor=sample_args.scale_factor, svg_filename = sample_args.filename+'.pdf.svg')
  return [strokes, params]

def sample_waves():
  [strokes, params] = model.sample(sess, sample_args.sample_length)
  # [strokes, params] = model.sample_seeded(sess, sample_args.sample_length)
  return [strokes, params]
    

# [strokes, params] = sample_stroke()
[strokes, params] = sample_waves()
print type(strokes), strokes.shape #, type(params), params

from scipy.io import wavfile
print(np.max(strokes[:,0:2]), np.min(strokes[:,0:2]))
# wavfile.write("explorers_MP3WRAP_sample.wav", 44100, strokes[:,0:2]/65535.)
wavname = "notypeinst_sample_%s.wav" % (time.strftime("%Y%m%d-%H%M%S"))
print "saving %s" % wavname
wavfile.write(wavname, 44100, strokes[:,0:2]/sample_args.scale_factor)
