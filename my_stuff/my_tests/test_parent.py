from foo import Animal

ME = Animal(flying=False)

class TestParent():

    def test_flying_is_false(self):
        assert ME.flying == False