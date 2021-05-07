from mrmustard._circuit import BaseCircuit
from mrmustard._opt import BaseOptimizer
from mrmustard._gates import BaseBSgate, BaseDgate, BaseSgate, BaseRgate, BaseGgate, BaseLoss
from mrmustard._backends.tfbackend import TFCircuitBackend, TFGateBackend, TFOptimizerBackend, TFMathbackend

class Circuit(BaseCircuit,TFCircuitBackend): pass
class Optimizer(TFOptimizerBackend, BaseOptimizer): pass
class Sgate(TFGateBackend, BaseSgate, TFMathbackend): pass
class Dgate(TFGateBackend, BaseDgate, TFMathbackend): pass
class Rgate(TFGateBackend, BaseRgate, TFMathbackend): pass
class Ggate(TFGateBackend, BaseGgate, TFMathbackend): pass
class BSgate(TFGateBackend, BaseBSgate, TFMathbackend): pass
class Loss(TFGateBackend, BaseLoss, TFMathbackend): pass
class S2gate(TFGateBackend, BaseSgate, TFMathbackend): pass
__all__ = ['Circuit', 'Optimizer', 'Sgate', 'Dgate', 'Ggate', 'BSgate', 'Rgate', 'Lossgate', 'S2gate']