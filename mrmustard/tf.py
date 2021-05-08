from mrmustard._circuit import BaseCircuit
from mrmustard._opt import BaseOptimizer
from mrmustard._gates import BaseGate, BaseBSgate, BaseDgate, BaseSgate, BaseRgate, BaseGgate, BaseLoss, BaseS2gate
from mrmustard._backends.tfbackend import TFCircuitBackend, TFGateBackend, TFOptimizerBackend, TFMathbackend

BaseGate._backend = TFMathbackend() # injecting tf math backend into all the gates

class Circuit(BaseCircuit,TFCircuitBackend): pass
class Optimizer(TFOptimizerBackend, BaseOptimizer): pass

class Sgate(TFGateBackend, BaseSgate): pass
class Dgate(TFGateBackend, BaseDgate): pass
class Rgate(TFGateBackend, BaseRgate): pass
class Ggate(TFGateBackend, BaseGgate): pass
class BSgate(TFGateBackend, BaseBSgate): pass
class LossChannel(TFGateBackend, BaseLoss): pass
class S2gate(TFGateBackend, BaseS2gate): pass

__all__ = ['Circuit', 'Optimizer', 'Sgate', 'Dgate', 'Ggate', 'BSgate', 'Rgate', 'LossChannel', 'S2gate']