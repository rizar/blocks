from blocks.bricks import Feedforward, Initializable, Tanh, Linear, application
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.bricks.parallel import Fork


class LanguageModel(Feedforward, Initializable):
    """The traditional recurrent transition.

    The most well-known recurrent transition: a matrix multiplication,
    optionally followed by a non-linearity.

    Parameters
    ----------
    dim : int
        The dimension of the hidden state
    activation : :class:`.Brick`
        The brick to apply as activation.

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    """
    def __init__(self, dim, transition=None, preaggregator=None, **kwargs):
        super(LanguageModel, self).__init__(**kwargs)
        if transition is None:
            transition = SimpleRecurrent(dim=dim, activation=Tanh())
        self.preaggregator = preaggregator
        self.transition = transition
        self.dim = dim
        if self.preaggregator is None:
            fork_input_dim = dim
        else:
            fork_input_dim = preaggregator.get_dim('output')
        self.fork = Fork([name for name in
                          transition.apply.sequences
                          if name != 'mask'],
                         fork_input_dim,
                         output_dims=[self.transition.get_dim(name) for name
                                      in transition.apply.sequences
                                      if name != 'mask'],
                         name='fork',
                         prototype=Linear(), **kwargs)
        self.children = [self.fork, self.transition]
        if self.preaggregator is not None:
            self.children.append(self.preaggregator)

    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name == 'lm_output':
            return self.dim
        if name == 'inputs':
            return self.dim
        return super(LanguageModel, self).get_dim(name)

    @application(inputs=['inputs', 'mask'], outputs=['lm_output'])
    def apply(self, inputs=None, mask=None):
        if self.preaggregator is not None:
            preaggregated = self.preaggregator.apply(inputs)
            forked = self.fork.apply(preaggregated, as_dict=True)
        else:
            forked = self.fork.apply(inputs, as_dict=True)
        return self.transition.apply(mask=mask, **forked)

    @application(inputs=['inputs', 'mask', 'lm_output'], outputs=['lm_output'])
    def apply_step(self, inputs=None, mask=None, lm_output=None):
        if self.preaggregator is not None:
            preaggregated = self.preaggregator.apply(inputs)
            forked = self.fork.apply(preaggregated, as_dict=True)
        else:
            forked = self.fork.apply(inputs, as_dict=True)
        return self.transition.apply(states=lm_output, mask=mask,
                                     iterate=False, **forked)

    @application
    def initial_state(self, name, batch_size, *args, **kwargs):
        if name == 'lm_output':
            return self.transition.initial_state('states', batch_size, *args,
                                                 **kwargs)

