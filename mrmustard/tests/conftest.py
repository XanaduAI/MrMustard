import os
import pytest
from hypothesis import settings, Verbosity

print("pytest.conf -----------------------")

settings.register_profile("ci", max_examples=1000, deadline=None)
settings.register_profile("dev", max_examples=10, deadline=None)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose, deadline=None)

settings.load_profile(os.getenv(u"HYPOTHESIS_PROFILE", "dev"))
