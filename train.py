import numpy as np
import tensorflow as tf
import pylab as pl

import argparse
import time
import os, sys
import cPickle

from utils import DataLoader
from model import Model
from modela import Model2Df
from modelb import ModelNDf

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--rnn_size', type=int, default=256,
                     help='size of RNN hidden state')
  parser.add_argument('--num_layers', type=int, default=2,
                     help='number of layers in the RNN')
  parser.add_argument('--model', type=str, default='lstm',
                     help='rnn, gru, or lstm')
  parser.add_argument('--modelfile', type=str, default=None,
                     help='Load a model file')
  parser.add_argument('--batch_size', type=int, default=50,
                     help='minibatch size')
  parser.add_argument('--seq_length', type=int, default=300,
                     help='RNN sequence length')
  parser.add_argument('--num_epochs', type=int, default=30,
                     help='number of epochs')
  parser.add_argument('--save_every', type=int, default=500,
                     help='save frequency')
  parser.add_argument('--grad_clip', type=float, default=10.,
                     help='clip gradients at this value')
  parser.add_argument('--learning_rate', type=float, default=0.005,
                     help='learning rate')
  parser.add_argument('--decay_rate', type=float, default=0.95,
                     help='decay rate for rmsprop')
  parser.add_argument('--num_mixture', type=int, default=20,
                     help='number of gaussian mixtures')
  parser.add_argument('--data_scale', type=float, default=20,
                     help='factor to scale raw data down by')
  parser.add_argument('--keep_prob', type=float, default=0.8,
                     help='dropout keep probability')
  parser.add_argument("--supmodel", default="2D")
  parser.add_argument("--datafile", type=str, default=None)
  args = parser.parse_args()
  train(args)

def train(args):
    print "loading data"
    data_loader = DataLoader(args.batch_size, args.seq_length, args.data_scale, limit=32768., datafilename = args.datafile)

    with open(os.path.join('save', 'config.pkl'), 'w') as f:
        cPickle.dump(args, f)

    print "creating model %s" % args.supmodel
    if args.supmodel == "2Deos":
        model = Model(args)
    elif args.supmodel == "2D":
        model = Model2Df(args)
        # model = ModelNDf(args)
    else:
        print "unknown supmodel"
        sys.exit()

    # import pylab as pl
    print "starting run"
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables(), max_to_keep = 100)
        offset_epochs = 0
        if args.modelfile != None:
            print "Loading model from checkpoint %s" % (args.modelfile)
            saver.restore(sess, args.modelfile)
            offset_epochs = int(args.modelfile.split("-")[-1])/data_loader.num_batches
            print "Setting epoch pointers to %d - %d" % (offset_epochs, offset_epochs+args.num_epochs)
        train_losses = []
        for e in xrange(offset_epochs, offset_epochs+args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            state = model.initial_state.eval()
            for b in xrange(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                # pl.plot(x[0])
                # pl.plot(y[0])
                # pl.show()
                # for i in range(len(x)):
                    # print "min, max xy", np.min(x[i]), np.max(x[i]), np.min(y[i]), np.max(y[i])
                    # print len(x), len(y), np.min(x[0]), np.max(x[0]), np.min(y[0]), np.max([y])
                feed = {model.input_data: x, model.target_data: y, model.initial_state: state}
                train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)
                end = time.time()
                # print
                train_losses.append(train_loss)
                print "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(e * data_loader.num_batches + b,
                            (offset_epochs + args.num_epochs) * data_loader.num_batches,
                            e, train_loss, end - start)
                if (e * data_loader.num_batches + b) % args.save_every == 0 and ((e * data_loader.num_batches + b) > 0):
                    print "min, max xy", np.min(x[-1]), np.max(x[-1]), np.min(y[-1]), np.max(y[-1])
                    checkpoint_path = os.path.join('save', 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches + b)
                    print "model saved to {}".format(checkpoint_path)
                    np.save("save/train_losses.npy", np.asarray(train_losses))
        pl.title("write-rnn-tf: train losses")
        pl.plot(train_losses)
        pl.show()
        np.save("save/train_losses.npy", np.asarray(train_losses))

if __name__ == '__main__':
  main()


