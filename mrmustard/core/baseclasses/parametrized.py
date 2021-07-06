from abc import ABC


class Parametrized(ABC):
    r"""
    Base class for all parametrized objects.

    Arguments (all keyword):
        For each supported parameter:
        xxx (tensor): initial parameter
        xxx_bounds (float, float): numerical bounds. Default is (None, None) for unbounded parameters.
        xxx_trainable (bool): whether the parameter `xxx` is trainable or not.
        yyy (any): other parameters
    """
    def __init__(self, **kwargs):
        self._trainable_parameters = []
        self._constant_parameters = []

        self.param_names = [key for key in kwargs if key + '_trainable' in kwargs]  # every parameter can be trainable! 🚀
        for name in self.param_names:
            if kwargs[name + "_trainable"]:
                param = self._math_backend.new_variable(kwargs[name], kwargs[name + "_bounds"], name)
                self._trainable_parameters.append(param)
            else:
                param = self._math_backend.new_constant(kwargs[name], name)
                self._constant_parameters.append(param)
            self.__dict__[name] = param  # making param available as gate.param
        for key, val in kwargs.items():
            if not any(word in key for word in self.param_names):
                self.__dict__['_' + key] = val  # making other values available as gate._val
