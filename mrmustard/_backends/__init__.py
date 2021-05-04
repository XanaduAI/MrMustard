from abc import ABC, abstractproperty

class SettingsInterface(ABC):
    symplectic_lr: float
    euclidean_lr: float
    double_precision: bool

    @abstractproperty
    def dtype(self): pass