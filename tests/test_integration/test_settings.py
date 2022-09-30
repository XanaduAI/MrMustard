from mrmustard import settings
import pytest


def test_backend_name():
    with pytest.raises(ValueError) as exception:
        settings.backend = 'BUMBUM'

    