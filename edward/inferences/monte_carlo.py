from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow as tf

from edward.inferences.inference import Inference
from edward.models import Empirical, RandomVariable
from edward.util import get_session


class MonteCarlo(Inference):
  """Base class for Monte Carlo inference methods.
  """
  def __init__(self, latent_vars, data=None, model_wrapper=None):
    """Initialization.

    Parameters
    ----------
    latent_vars : list of RandomVariable or
                  dict of RandomVariable to RandomVariable
      Collection of random variables to perform inference on. If
      list, each random variable will be implictly approximated
      using a ``Empirical`` random variable that is defined
      internally (with support matching each random variable).
      If dictionary, each random variable must be a ``Empirical``
      random variable.
    data : dict, optional
      Data dictionary which binds observed variables (of type
      `RandomVariable`) to their realizations (of type `tf.Tensor`).
      It can also bind placeholders (of type `tf.Tensor`) used in the
      model to their realizations.
    model_wrapper : ed.Model, optional
      A wrapper for the probability model. If specified, the random
      variables in `latent_vars`' dictionary keys are strings used
      accordingly by the wrapper. `data` is also changed. For
      TensorFlow, Python, and Stan models, the key type is a string;
      for PyMC3, the key type is a Theano shared variable. For
      TensorFlow, Python, and PyMC3 models, the value type is a NumPy
      array or TensorFlow tensor; for Stan, the value type is the
      type according to the Stan program's data block.

    Examples
    --------
    Most explicitly, MonteCarlo is specified via a dictionary:

    >>> qpi = Empirical(params=tf.Variable(tf.zeros(K-1)))
    >>> qmu = Empirical(params=tf.Variable(tf.zeros(K*D)))
    >>> qsigma = Empirical(params=tf.Variable(tf.zeros(K*D)))
    >>> MonteCarlo({pi: qpi, mu: qmu, sigma: qsigma}, data)

    We also automate the specification of ``Empirical`` random
    variables, so one can pass in a list of latent variables instead:

    >>> MonteCarlo([beta], data)
    >>> MonteCarlo([pi, mu, sigma], data)

    It defaults to Empirical random variables with 10,000 samples.
    However, for model wrappers, lists are not supported, e.g.,

    >>> MonteCarlo(['z'], data, model_wrapper)

    This is because internally with model wrappers, we have no way of
    knowing the dimensions in which to infer each latent variable. One
    must explicitly pass in the Empirical random variables.

    Notes
    -----
    The number of Monte Carlo iterations is set according to the
    minimum of all Empirical sizes.
    """
    if isinstance(latent_vars, list):
      with tf.variable_scope("posterior"):
        if model_wrapper is None:
          latent_vars = {rv: Empirical(params=tf.Variable(
              tf.zeros([1e4] + rv.get_batch_shape().as_list())))
              for rv in latent_vars}
        else:
          raise NotImplementedError("A list is not supported for model "
                                    "wrappers. See documentation.")
    elif isinstance(latent_vars, dict):
      for qz in six.itervalues(latent_vars):
        if not isinstance(qz, Empirical):
          raise TypeError("Posterior approximation must consist of only "
                          "Empirical random variables.")

    super(MonteCarlo, self).__init__(latent_vars, data, model_wrapper)

  def initialize(self, *args, **kwargs):
    self.t = tf.Variable(0, trainable=False)
    self.n_accept = tf.Variable(0, trainable=False)
    self.train = self.build_update()

    min_t = np.amin([
        qz.distribution.n for qz in six.itervalues(self.latent_vars)])
    kwargs['n_iter'] = min_t - 1
    return super(MonteCarlo, self).initialize(*args, **kwargs)

  def update(self, feed_dict=None):
    """Run one iteration of sampling for Monte Carlo.

    Parameters
    ----------
    feed_dict : dict, optional
      Feed dictionary for a TensorFlow session run. It is used to feed
      placeholders that are not fed during initialization.

    Returns
    -------
    dict
      Dictionary of algorithm-specific information. In this case, the
      loss function value after one iteration.
    """
    if feed_dict is None:
      feed_dict = {}

    for key, value in six.iteritems(self.data):
      if not isinstance(key, RandomVariable) and not isinstance(key, str):
        feed_dict[key] = value

    sess = get_session()
    sess.run(self.train, feed_dict)
    accept_rate = sess.run(self.n_accept / self.t)
    return {'accept_rate': accept_rate}

  def print_progress(self, t, info_dict):
    """Print progress to output.
    """
    if self.n_print is not None:
      if t % self.n_print == 0:
        accept_rate = info_dict['accept_rate']
        print("iter {:d} accept rate {:.2f}".format(t, accept_rate))
        for rv in six.itervalues(self.latent_vars):
          print(rv)
          std = rv.std().eval()
          print("std: \n" + std.__str__())

  def build_update(self):
    """Build update, which returns an assign op for parameters in
    the Empirical random variables.

    Any derived class of ``MonteCarlo`` **must** implement
    this method.

    Raises
    ------
    NotImplementedError
    """
    raise NotImplementedError()
