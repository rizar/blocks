"""Defines models.

A model is a thin layer of abstraction between the user-defined computation
graph, bricks, parameters and main loop extensions.

"""
import logging
from collections import OrderedDict
from itertools import chain

from blocks.graph import ComputationGraph
from blocks.select import Selector
from blocks.filter import VariableFilter, get_brick
from blocks.roles import OBJECTIVE

logger = logging.getLogger()


class BrickComputationGraph(ComputationGraph):

    def __init__(self, outputs):
        super(BrickComputationGraph, self).__init__(outputs)
        self._get_bricks()

    def _get_bricks(self):
        bricks = [get_brick(var) for var
                  in self.variables + self.scan_variables if get_brick(var)]
        children = set(chain(*(brick.children for brick in bricks)))
        # Quadratic complexity: we should not have thousands of
        # top-level bricks.
        self.top_bricks = []
        for brick in bricks:
            if brick not in children and brick not in self.top_bricks:
                self.top_bricks.append(brick)
        if len(set(b.name for b in self.top_bricks)) < len(self.top_bricks):
            raise ValueError("top bricks with the same name")


class Model(object):

    def __init__(self, outputs=None,
                 computation_graph=None, objective=None,
                 parameters=None, top_bricks=None):
        self._computation_graph = computation_graph
        self._objective = objective
        self._parameters = parameters
        self._top_bricks = top_bricks
        if outputs:
            if self._computation_graph:
                raise ValueError
            self._computation_graph = BrickComputationGraph(outputs)

    def get_computation_graph(self):
        if not self._computation_graph:
            raise ValueError
        return self._computation_graph

    def set_computation_graph(self, computation_graph):
        self._computation_graph = computation_graph

    def get_objective(self):
        if self._objective:
            return self._objective
        computation_graph = self.get_computation_graph()
        if len(computation_graph.outputs) == 1:
            return computation_graph.outputs[0]
        objective, = VariableFilter(
            roles=[OBJECTIVE])(computation_graph)
        raise ValueError

    def set_objective(self, objective):
        self._objective = objective

    def get_top_bricks(self):
        if self._top_bricks:
            return self._top_bricks
        if isinstance(self.get_computation_graph(), BrickComputationGraph):
            return self.get_computation_graph().top_bricks
        raise ValueError

    def set_top_bricks(self, top_bricks):
        self._top_brikcs = top_bricks

    def get_parameters(self, hierarchical_names=False):
        if self._parameters:
            parameters = self._parameters

        parameters = self.get_computation_graph().parameters
        if hierarchical_names:
            param2name = {
                v: k for k, v in
                Selector(self.get_top_bricks()).get_params().items()}
            return OrderedDict(
                [(param2name[p] if p in param2name else p.name, p)
                 for p in parameters])
        return parameters

    def set_parameters(self, parameters):
        self._parameters = parameters

    def get_param_values(self):
        """Return the values of model parameters.

        Returns
        -------
        param_values : OrderedDict
            Dictionary of (parameter name, :class:`~numpy.ndarray`) pairs.

        """
        return OrderedDict(
            (name, param.get_value())
            for name, param
            in self.get_parameters(hierarchical_names=True).items())

    def set_param_values(self, param_values):
        """Set the values of model parameters.

        Parameters
        ----------
        param_values : OrderedDict
            Dictionary of (parameter name, :class:`~numpy.ndarray`) pairs.

        """
        params = self.get_parameters(hierarchical_names=True)

        unknown = set(param_values) - set(params)
        missing = set(params) - set(param_values)
        if len(unknown):
            logger.error("unknown parameter names: {}\n".format(unknown))
        if len(missing):
            logger.error("missing values for parameters: {}\n".format(missing))

        for name, value in param_values.items():
            if name in params:
                params[name].set_value(value)
