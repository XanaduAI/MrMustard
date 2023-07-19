class Animal():
    def __init__(self, flying:bool = False) -> None:
        self.flying = flying

    def answer(self):
        return 1954



class Bird(Animal):
    def __init__(self, flying: bool = True) -> None:
        super().__init__(flying)

    def answer(self):
        return 42