from foo import Bird
from test_parent import TestParent

ME = Bird(flying=False)

class TestChild(TestParent):

    def test_flying_is_true(self):
        assert ME.flying == True

    def test_knows_answer_to_the_universe(self):
        assert ME.answer() == 42