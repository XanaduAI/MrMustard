if __name__ == "mrmustard.tf":
    from mrmustard._circuit import Circuit
    from mrmustard._opt import BaseOptimizer
    from mrmustard._detectors import Detector, PNR, APD
    from mrmustard._gates import Gate, BSgate, Sgate, Rgate, Dgate, Ggate, LossChannel, S2gate
    from mrmustard._states import Vacuum
    from mrmustard._states import State
    from mrmustard._backends.tfbackend import (
        TFGateBackend,
        TFOptimizerBackend,
        TFStateBackend,
        TFMathBackend,
    )

    math = TFMathBackend()

    Detector._math_backend = math
    Circuit._math_backend = math
    Gate._math_backend = math
    Gate._gate_backend = TFGateBackend()
    State._math_backend = math
    State._state_backend = TFStateBackend()

    class Optimizer(TFOptimizerBackend, BaseOptimizer):
        pass

    __all__ = [
        "BSgate",
        "Sgate",
        "Rgate",
        "Dgate",
        "Ggate",
        "LossChannel",
        "S2gate",
        "PNR",
        "APD",
        "Circuit",
        "Optimizer",
        "Vacuum",
    ]
