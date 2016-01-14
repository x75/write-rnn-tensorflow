import tensorflow as tf
from tensorflow.models.rnn import rnn_cell, rnn
from tensorflow.models.rnn import seq2seq

import numpy as np
import random

class Model2Df(): # two-dimensional data (stereo), float
  def __init__(self, args, infer=False):
    self.args = args

    self.dim = 2
    
    if infer:
      args.batch_size = 1
      args.seq_length = 1

    if args.model == 'rnn':
      cell_fn = rnn_cell.BasicRNNCell
    elif args.model == 'gru':
      cell_fn = rnn_cell.GRUCell
    elif args.model == 'lstm':
      cell_fn = rnn_cell.BasicLSTMCell
    elif args.model == "lstmp":
      cell_fn = rnn_cell.LSTMCell
    elif args.model == "cw":
      cell_fn = rnn_cell.CWRNNCell
    else:
      raise Exception("model type not supported: {}".format(args.model))

    if args.model == "lstmp":
      cell = cell_fn(args.rnn_size, self.dim, use_peepholes=True, num_proj=args.rnn_size)
    elif args.model == "lstm":
      cell = cell_fn(args.rnn_size, forget_bias = 5.0)
    elif args.model == "cw":
      cell = cell_fn(args.rnn_size, [1, 4, 16, 64])
    else:
      cell = cell_fn(args.rnn_size)

    cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)

    if (infer == False and args.keep_prob < 1): # training mode
      cell = rnn_cell.DropoutWrapper(cell, output_keep_prob = args.keep_prob)

    self.cell = cell

    self.input_data = tf.placeholder(dtype=tf.float32, shape=[None, args.seq_length, self.dim])
    self.target_data = tf.placeholder(dtype=tf.float32, shape=[None, args.seq_length, self.dim])
    self.initial_state = cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)

    self.num_mixture = args.num_mixture
    NOUT = self.num_mixture * 3 * self.dim # i/o dim * (pi + mu + sig) #  + corr

    with tf.variable_scope('rnnlm'):
      output_w = tf.get_variable("output_w", [args.rnn_size, NOUT])
      output_b = tf.get_variable("output_b", [NOUT])

    inputs = tf.split(1, args.seq_length, self.input_data)
    inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

    outputs, states = seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=None, scope='rnnlm')
    # outputs, states = rnn.rnn(cell, inputs, self.initial_state, sequence_length = args.seq_length, scope='rnnlm')
    output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
    output = tf.nn.xw_plus_b(output, output_w, output_b)
    self.final_state = states[-1]

    # reshape target data so that it is compatible with prediction shape
    flat_target_data = tf.reshape(self.target_data,[-1, self.dim])
    # [x1_data, x2_data, eos_data] = tf.split(1, 3, flat_target_data)
    [x1_data, x2_data] = tf.split(1, self.dim, flat_target_data) # FIXME: 2d hardcoded

    # long method:
    #flat_target_data = tf.split(1, args.seq_length, self.target_data)
    #flat_target_data = [tf.squeeze(flat_target_data_, [1]) for flat_target_data_ in flat_target_data]
    #flat_target_data = tf.reshape(tf.concat(1, flat_target_data), [-1, 3])

    def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
      # eq # 24 and 25 of http://arxiv.org/abs/1308.0850
      norm1 = tf.sub(x1, mu1)
      norm2 = tf.sub(x2, mu2)
      s1s2 = tf.mul(s1, s2)
      z = tf.square(tf.div(norm1, s1))+tf.square(tf.div(norm2, s2))-2*tf.div(tf.mul(rho, tf.mul(norm1, norm2)), s1s2)
      negRho = 1-tf.square(rho)
      result = tf.exp(tf.div(-z,2*negRho))
      denom = 2*np.pi*tf.mul(s1s2, tf.sqrt(negRho))
      result = tf.div(result, denom)
      return result

    def get_lossfunc(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, x1_data, x2_data):
      result0 = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr)
      # implementing eq # 26 of http://arxiv.org/abs/1308.0850
      epsilon = 1e-20
      result1 = tf.mul(result0, z_pi)
      result1 = tf.reduce_sum(result1, 1, keep_dims=True)
      result1 = -tf.log(tf.maximum(result1, epsilon)) # at the beginning, some errors are exactly zero.

      # result2 = tf.mul(z_eos, eos_data) + tf.mul(1-z_eos, 1-eos_data)
      # result2 = -tf.log(result2)

      result = result1 # + result2
      return tf.reduce_sum(result)

    # below is where we need to do MDN splitting of distribution params
    def get_mixture_coef(output):
      # returns the tf slices containing mdn dist params
      # ie, eq 18 -> 23 of http://arxiv.org/abs/1308.0850
      z = output
      # z_eos = z[:, 0:1]
      z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(1, self.dim * 3, z[:, 0:])

      # process output z's into MDN paramters

      # end of stroke signal
      # z_eos = tf.sigmoid(z_eos) # should be negated, but doesn't matter.

      # softmax all the pi's:
      max_pi = tf.reduce_max(z_pi, 1, keep_dims=True)
      z_pi = tf.sub(z_pi, max_pi)
      z_pi = tf.exp(z_pi)
      normalize_pi = tf.inv(tf.reduce_sum(z_pi, 1, keep_dims=True))
      z_pi = tf.mul(normalize_pi, z_pi)

      # exponentiate the sigmas and also make corr between -1 and 1.
      z_sigma1 = tf.exp(z_sigma1)
      z_sigma2 = tf.exp(z_sigma2)
      z_corr = tf.tanh(z_corr)

      return [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr]

    [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr] = get_mixture_coef(output)

    self.pi = o_pi
    self.mu1 = o_mu1
    self.mu2 = o_mu2
    self.sigma1 = o_sigma1
    self.sigma2 = o_sigma2
    self.corr = o_corr
    # self.eos = o_eos

    lossfunc = get_lossfunc(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, x1_data, x2_data)
    self.cost = lossfunc / (args.batch_size * args.seq_length)

    self.lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)
    optimizer = tf.train.AdamOptimizer(self.lr)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars))


  def sample(self, sess, num=1200):

    def get_pi_idx(x, pdf):
      N = pdf.size
      accumulate = 0
      for i in range(0, N):
        accumulate += pdf[i]
        if (accumulate >= x):
          return i
      print 'error with sampling ensemble'
      return -1

    def sample_gaussian_2d(mu1, mu2, s1, s2, rho):
      mean = [mu1, mu2]
      cov = [[s1*s1, rho*s1*s2], [rho*s1*s2, s2*s2]]
      x = np.random.multivariate_normal(mean, cov, 1)
      return x[0][0], x[0][1]

    prev_x = np.zeros((1, 1, self.dim), dtype=np.float32)
    # prev_x[0, 0, 2] = 1 # initially, we want to see beginning of new stroke
    prev_state = sess.run(self.cell.zero_state(1, tf.float32))

    strokes = np.zeros((num, self.dim), dtype=np.float32)
    mixture_params = []

    for i in xrange(num):

      feed = {self.input_data: prev_x, self.initial_state:prev_state}

      [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, next_state] = sess.run([self.pi, self.mu1, self.mu2, self.sigma1, self.sigma2, self.corr, self.final_state],feed)

      idx = get_pi_idx(random.random(), o_pi[0])

      # eos = 1 if random.random() < o_eos[0][0] else 0

      next_x1, next_x2 = sample_gaussian_2d(o_mu1[0][idx], o_mu2[0][idx], o_sigma1[0][idx], o_sigma2[0][idx], o_corr[0][idx])

      strokes[i,:] = [next_x1, next_x2] # FIXME: 2d hardcoded

      params = [o_pi[0], o_mu1[0], o_mu2[0], o_sigma1[0], o_sigma2[0], o_corr[0]]
      mixture_params.append(params)

      prev_x = np.zeros((1, 1, self.dim), dtype=np.float32)
      prev_x[0][0] = np.array([next_x1, next_x2], dtype=np.float32) # FIXME: 2d hardcoded
      prev_state = next_state
      if i % 100 == 0:
        print i, # feed

    strokes[:,0:2] *= self.args.data_scale # FIXME: 2d hardcoded
    return strokes, mixture_params

  def sample_seeded(self, sess, num=1200):

    def get_pi_idx(x, pdf):
      N = pdf.size
      accumulate = 0
      for i in range(0, N):
        accumulate += pdf[i]
        if (accumulate >= x):
          return i
      print 'error with sampling ensemble'
      return -1

    def sample_gaussian_2d(mu1, mu2, s1, s2, rho):
      mean = [mu1, mu2]
      cov = [[s1*s1, rho*s1*s2], [rho*s1*s2, s2*s2]]
      x = np.random.multivariate_normal(mean, cov, 1)
      return x[0][0], x[0][1]


    # seed
    print "seeding"
    prev_state = sess.run(self.cell.zero_state(1, tf.float32))
    for i in range(1000):
      # prev_x = np.random.uniform(-0.5, 0.5, (1, 1, self.dim)).astype(np.float32)
      prev_x = np.zeros((1, 1, self.dim), dtype=np.float32)
      s = np.sin(i/100.)
      prev_x[0, 0, 0:2] = s
      feed = {self.input_data: prev_x, self.initial_state:prev_state}
      [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, next_state] = sess.run([self.pi, self.mu1, self.mu2, self.sigma1, self.sigma2, self.corr, self.final_state],feed)
      prev_state = next_state
        
    print "rolling"
    # prev_x = np.zeros((1, 1, self.dim), dtype=np.float32)
    # prev_state = sess.run(self.cell.zero_state(1, tf.float32))
    prev_x = np.random.uniform(-0.5, 0.5, (1, 1, self.dim)).astype(np.float32)
    prev_state = next_state
    # prev_x[0, 0, 2] = 1 # initially, we want to see beginning of new stroke

    strokes = np.zeros((num, self.dim), dtype=np.float32)
    mixture_params = []

    for i in xrange(num):

      feed = {self.input_data: prev_x, self.initial_state:prev_state}

      [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, next_state] = sess.run([self.pi, self.mu1, self.mu2, self.sigma1, self.sigma2, self.corr, self.final_state],feed)

      idx = get_pi_idx(random.random(), o_pi[0])

      # eos = 1 if random.random() < o_eos[0][0] else 0

      next_x1, next_x2 = sample_gaussian_2d(o_mu1[0][idx], o_mu2[0][idx], o_sigma1[0][idx], o_sigma2[0][idx], o_corr[0][idx])

      strokes[i,:] = [next_x1, next_x2] # FIXME: 2d hardcoded

      params = [o_pi[0], o_mu1[0], o_mu2[0], o_sigma1[0], o_sigma2[0], o_corr[0]]
      mixture_params.append(params)

      prev_x = np.zeros((1, 1, self.dim), dtype=np.float32)
      prev_x[0][0] = np.array([next_x1, next_x2], dtype=np.float32) # FIXME: 2d hardcoded
      prev_state = next_state
      if i % 100 == 0:
        print i, # feed


    strokes[:,0:2] *= self.args.data_scale # FIXME: 2d hardcoded
    return strokes, mixture_params
