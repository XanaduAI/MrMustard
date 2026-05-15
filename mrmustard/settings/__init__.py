import sys

from .settings import Settings

sys.modules[__name__] = Settings()
