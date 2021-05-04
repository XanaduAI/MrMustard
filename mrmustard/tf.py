from mrmustard._circuit import BaseCircuit
from mrmustard._opt import BaseOptimizer
from mrmustard._gates import BaseBSgate, BaseDgate, BaseSgate, BaseRgate, BaseGgate, BaseLoss
from mrmustard._backends.tfbackend import TFCircuitBackend, TFGateBackend, TFOptimizerBackend

class Circuit(BaseCircuit,TFCircuitBackend): pass
class Optimizer(TFOptimizerBackend, BaseOptimizer): pass
class Sgate(TFGateBackend, BaseSgate): pass
class Dgate(TFGateBackend, BaseDgate): pass
class Ggate(TFGateBackend, BaseGgate): pass
class BSgate(TFGateBackend, BaseBSgate): pass
class Loss(TFGateBackend, BaseLoss): pass

__all__ = ['Circuit', 'Optimizer', 'Sgate', 'Dgate', 'Ggate', 'BSgate', 'Lossgate']