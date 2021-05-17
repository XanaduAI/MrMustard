if __name__ == "mrmustard.tf":
    from mrmustard._circuit import BaseCircuit
    from mrmustard._opt import BaseOptimizer
    from mrmustard._gates import Gate, BSgate, Sgate, Rgate, Dgate, Ggate, LossChannel, S2gate
    from mrmustard._backends.tfbackend import (
        TFCircuitBackend,
        TFGateBackend,
        TFOptimizerBackend,
        TFStateBackend,
        TFMathbackend,
    )
    from mrmustard._states import State

    Gate._math_backend = TFMathbackend()  # injecting tf math backend into all the gates
    Gate._gate_backend = TFGateBackend()  # injecting tf gate backend into all the gates
    State._state_backend = TFStateBackend()  # injecting tf state backend into all the states

    class Circuit(BaseCircuit, TFCircuitBackend):
        pass

    class Optimizer(TFOptimizerBackend, BaseOptimizer):
        pass
