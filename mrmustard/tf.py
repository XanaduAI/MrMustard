if __name__ == "mrmustard.tf":
    from mrmustard._circuit import BaseCircuit
    from mrmustard._opt import BaseOptimizer
    from mrmustard._detectors import Detector, PNR
    from mrmustard._gates import Gate, BSgate, Sgate, Rgate, Dgate, Ggate, LossChannel, S2gate
    from mrmustard._backends.tfbackend import (
        TFCircuitBackend,
        TFGateBackend,
        TFOptimizerBackend,
        TFStateBackend,
        TFMathbackend,
        TFDetectorBackend,
    )
    from mrmustard._states import State

    Gate._math_backend = TFMathbackend()
    Gate._gate_backend = TFGateBackend()
    State._state_backend = TFStateBackend()
    Detector._detector_backend = TFDetectorBackend()
    Detector._math_backend = TFMathbackend()
    BaseCircuit._math_backend = TFMathbackend()

    class Circuit(BaseCircuit, TFCircuitBackend):
        pass

    class Optimizer(TFOptimizerBackend, BaseOptimizer):
        pass
