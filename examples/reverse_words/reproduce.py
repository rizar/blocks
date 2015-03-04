#!/usr/bin/env python

from __future__ import print_function
import logging
import pprint
import numpy

import theano
from theano import tensor

from blocks.bricks import Tanh, Initializable
from blocks.bricks.base import application
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import SimpleRecurrent, Bidirectional
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.parallel import Fork
from blocks.bricks.sequence_generators import (
    SequenceGenerator, LinearReadout, SoftmaxEmitter, LookupFeedback)
from blocks.config_parser import config
from blocks.graph import ComputationGraph
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.filter import VariableFilter
from blocks.utils import named_copy, dict_union

config.recursion_limit = 100000
floatX = theano.config.floatX
logger = logging.getLogger(__name__)

# Dictionaries
all_chars = ([chr(ord('a') + i) for i in range(26)] +
             [chr(ord('0') + i) for i in range(10)] +
             [',', '.', '!', '?', '<UNK>'] +
             [' ', '<S>', '</S>'])
code2char = dict(enumerate(all_chars))
char2code = {v: k for k, v in code2char.items()}


class WordReverser(Initializable):
    """The top brick.

    It is often convenient to gather all bricks of the model under the
    roof of a single top brick.

    """
    def __init__(self, dimension, alphabet_size, **kwargs):
        super(WordReverser, self).__init__(**kwargs)
        encoder = Bidirectional(
            SimpleRecurrent(dim=dimension, activation=Tanh()))
        fork = Fork([name for name in encoder.prototype.apply.sequences
                    if name != 'mask'])
        fork.input_dim = dimension
        fork.output_dims = {name: dimension for name in fork.input_names}
        lookup = LookupTable(alphabet_size, dimension)
        transition = SimpleRecurrent(
            activation=Tanh(),
            dim=dimension, name="transition")
        attention = SequenceContentAttention(
            state_names=transition.apply.states,
            sequence_dim=2 * dimension, match_dim=dimension, name="attention")
        readout = LinearReadout(
            readout_dim=alphabet_size, source_names=["states"],
            emitter=SoftmaxEmitter(name="emitter"),
            feedback_brick=LookupFeedback(alphabet_size, dimension),
            name="readout")
        generator = SequenceGenerator(
            readout=readout, transition=transition, attention=attention,
            name="generator")

        self.lookup = lookup
        self.fork = fork
        self.encoder = encoder
        self.generator = generator
        self.children = [lookup, fork, encoder, generator]

    @application
    def cost(self, chars, chars_mask, targets, targets_mask):
        return self.generator.cost(
            targets, targets_mask,
            attended=self.encoder.apply(
                **dict_union(
                    self.fork.apply(self.lookup.lookup(chars), as_dict=True),
                    mask=chars_mask)),
            attended_mask=chars_mask)

    @application
    def generate(self, chars):
        return self.generator.generate(
            n_steps=3 * chars.shape[0], batch_size=chars.shape[1],
            attended=self.encoder.apply(
                **dict_union(self.fork.apply(self.lookup.lookup(chars),
                             as_dict=True))),
            attended_mask=tensor.ones(chars.shape))


reverser = WordReverser(100, len(char2code), name="reverser")

# Initialization settings
reverser.weights_init = IsotropicGaussian(0.1)
reverser.biases_init = Constant(0.0)
reverser.push_initialization_config()
reverser.encoder.weghts_init = Orthogonal()
reverser.generator.transition.weights_init = Orthogonal()

# Build the cost computation graph
chars = tensor.lmatrix("features")
chars_mask = tensor.matrix("features_mask")
targets = tensor.lmatrix("targets")
targets_mask = tensor.matrix("targets_mask")
batch_cost = reverser.cost(
    chars, chars_mask, targets, targets_mask).sum()
batch_size = named_copy(chars.shape[1], "batch_size")
cost = aggregation.mean(batch_cost,  batch_size)
cost.name = "sequence_log_likelihood"
logger.info("Cost graph is built")

# Give an idea of what's going on
model = Model(cost)
params = model.get_params()
logger.info("Parameters:\n" +
            pprint.pformat(
                [(key, value.get_value().shape) for key, value
                    in params.items()],
                width=120))

# Initialize parameters
for brick in model.get_top_bricks():
    brick.initialize()

cg = ComputationGraph(cost)
r = reverser
weights, = VariableFilter(bricks=[r.generator], name="weights")(cg)
weight_sum = named_copy(weights.sum(), "weight_sum")
data = {chars: numpy.ones((3, 4), dtype='int64'),
        chars_mask: numpy.ones((3, 4)).astype(floatX),
        targets: numpy.ones((5, 4), dtype='int64'),
        targets_mask: numpy.ones((5, 4)).astype(floatX)}
print("Must be 20: ", weight_sum.eval(data))
