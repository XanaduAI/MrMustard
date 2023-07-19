

# Object classes
class Animal():
    def __init__(self, flying:bool) -> None:
        self.flying = flying

class Bird(Animal):
    def __init__(self, flying: bool) -> None:
        super().__init__(flying)


# Defining the object we want to test over
ME = Bird(flying=True)


# Test classes
class TestParent():

    def test_flying_is_false(self):
        assert ME.flying == False


class TestChild():

    def test_flying_is_true(self):
        assert ME.flying == True
