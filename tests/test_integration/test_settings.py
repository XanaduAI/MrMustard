from mrmustard import settings
import pytest


def test_backend_name():
    settings.backend = 'Tensorflow'
    settings.backend = 'torch'
    with pytest.raises(ValueError) as exception:
        settings.backend = "BUMBUM"
