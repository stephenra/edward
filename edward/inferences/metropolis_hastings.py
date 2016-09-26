from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.monte_carlo import MonteCarlo
from edward.models import RandomVariable, Uniform
from edward.util import copy


class MetropolisHastings(MonteCarlo):
  """Metropolis-Hastings.
  """
  def __init__(self, latent_vars, proposal_vars, data=None, model_wrapper=None):
    """
    Parameters
    ----------
    proposal_vars : dict of RandomVariable to RandomVariable
      Collection of random variables to perform inference on; each is
      binded to a proposal distribution p(z' | z).

    Examples
    --------
    >>> z = Normal(mu=0.0, sigma=1.0)
    >>> x = Normal(mu=tf.ones(10) * z, sigma=1.0)
    >>>
    >>> qz = Empirical(tf.Variable(tf.zeros([500])))
    >>> proposal_z = Normal(mu=z, sigma=0.5)
    >>> data = {x: np.array([0.0] * 10, dtype=np.float32)}
    >>> inference = ed.MetropolisHastings({z: qz}, {z: proposal_z}, data)

    Notes
    -----
    Initialization is assumed from params[0, :]. This generalizes
    initializing randomly and initializing from user input. Updates
    are along this outer dimension, where iteration t updates
    params[t, :] in each Empirical random variable.

    No warm-up is implemented. Users must run MCMC for a long period
    of time, then manually burn in the Empirical random variable.

    The updates assume each Empirical random variable is directly
    parameterized by tf.Variables().
    """
    self.proposal_vars = proposal_vars
    super(MetropolisHastings, self).__init__(latent_vars, data, model_wrapper)

  def build_update(self):
    """
    Draw sample from proposal conditional on last sample. Then accept
    or reject the sample based on the ratio,

    ratio = log p(x, znew) - log p(x, zold) +
            log g(znew | zold) - log g(zold | znew)
          = sum_z [ log p(znew) - log p(zold) +
                    log g(znew | zold) - log g(zold | znew) ] +
            sum_x [ log p(x | znew) - sum_x log p(x | zold) ]
    """
    old_sample = {z: tf.gather(qz.distribution.params, self.t)
                  for z, qz in six.iteritems(self.latent_vars)}

    # Draw proposed sample and calculate acceptance ratio.
    new_sample = {}
    ratio = 0.0
    for z, proposal_z in six.iteritems(self.proposal_vars):
      # Build proposal g(znew | zold).
      proposal_znew = copy(proposal_z, old_sample, scope='proposal_znew')
      # Sample znew ~ g(znew | zold).
      new_sample[z] = proposal_z.sample()
      # Increment ratio.
      ratio += tf.reduce_sum(z.log_prob(new_sample[z]))
      ratio -= tf.reduce_sum(z.log_prob(old_sample[z]))
      ratio += tf.reduce_sum(proposal_znew.log_prob(new_sample[z]))

    for z, proposal_z in six.iteritems(self.proposal_vars):
      # Build proposal p(zold | znew).
      proposal_zold = copy(proposal_z, new_sample, scope='proposal_zold')
      # Increment ratio.
      ratio -= tf.reduce_sum(proposal_zold.log_prob(old_sample[z]))

    for x, obs in six.iteritems(self.data):
      if isinstance(x, RandomVariable):
        # Build likelihood p(x | znew).
        x_znew = copy(x, new_sample, scope='x_znew')
        # Build likelihood p(x | zold).
        x_zold = copy(x, old_sample, scope='x_zold')
        # Increment ratio.
        ratio += tf.reduce_sum(x_znew.log_prob(obs))
        ratio -= tf.reduce_sum(x_zold.log_prob(obs))

    # Accept or reject sample.
    u = Uniform().sample()
    accept = tf.log(u) < ratio
    sample_values = tf.cond(accept, lambda: list(six.itervalues(new_sample)),
                                    lambda: list(six.itervalues(old_sample)))
    if not isinstance(sample_values, list):
      # ``tf.cond`` returns tf.Tensor if output is a list of size 1.
      sample_values = [sample_values]

    sample = {z: sample_value for z, sample_value in
              zip(six.iterkeys(new_sample), sample_values)}

    # Update Empirical random variables.
    assign_ops = []
    variables = {x.name: x for x in
                 tf.get_default_graph().get_collection(tf.GraphKeys.VARIABLES)}
    for z, qz in six.iteritems(self.latent_vars):
      variable_read = qz.distribution.params.op.inputs[0]
      variable = variables[variable_read.op.inputs[0].name]
      assign_ops.append(tf.scatter_update(variable, self.t + 1, sample[z]))

    # Increment t.
    assign_ops.append(self.t.assign_add(1))
    # Increment n_accept (if accepted).
    assign_ops.append(self.n_accept.assign_add(tf.select(accept, 1, 0)))
    return tf.group(*assign_ops)
