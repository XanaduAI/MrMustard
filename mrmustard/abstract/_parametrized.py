from abc import ABC
from mrmustard import TrainPlugin

class Parametrized(ABC):
    r"""
    Abstract base class for all parametrized objects (gates, detectors, etc...)

    Arguments (must be all called with keyword):
        For each trainable parameter:
        xxx (tensor): initial value
        xxx_bounds (float, float): numerical bounds. Default is (None, None) for unbounded parameters
        xxx_trainable (bool): whether the parameter `xxx` will be optimized
        yyy (any): other parameters
    """

    _train = TrainPlugin()

    def __init__(self, **kwargs):
        self._trainable_parameters = []
        self._constant_parameters = []
        self.param_names = [key for key in kwargs if key + "_trainable" in kwargs]  # every parameter can be trainable! 🚀

        for name in self.param_names:
            if kwargs[name + "_trainable"]:
                var = self._train.new_variable(kwargs[name], kwargs[name + "_bounds"], name)
                self._trainable_parameters.append(var)
                self.__dict__[name] = var  # making params available as gate.param
            else:
                const = self._train.new_constant(kwargs[name], name)
                self._constant_parameters.append(const)
                self.__dict__[name] = const
        for key, val in kwargs.items():
            if not any(word in key for word in self.param_names):
                self.__dict__["_" + key] = val  # making other values available as gate._val

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # override the necessary properties in the concrete subclasses:
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @property
    def is_symplectic(self):
        return False

    @property
    def is_orthogonal(self):
        return False

    @property
    def is_euclidean(self):
        return False
    