class TagDispenser:
    r"""A singleton class that generates unique tags (ints).
    It can be given back tags to reuse them.

    Example:
        >>> dispenser = TagDispenser()
        >>> dispenser.get_tag()
        0
        >>> dispenser.get_tag()
        1
        >>> dispenser.give_back_tag(0)
        >>> dispenser.get_tag()
        0
        >>> dispenser.get_tag()
        2
    """
    _instance = None
    _tags = []
    _counter = 0

    def __new__(cls):
        if TagDispenser._instance is None:
            TagDispenser._instance = object.__new__(cls)
        return TagDispenser._instance

    def get_tag(self) -> int:
        """Returns a new unique tag."""
        if len(self._tags) > 0:
            return self._tags.pop(0)
        else:
            self._counter += 1
            return self._counter - 1

    def give_back_tag(self, tag: int):
        """Gives back a tag to be reused."""
        if isinstance(tag, int) and tag not in self._tags and tag < self._counter:
            self._tags.append(tag)
        else:
            raise ValueError(f"Cannot accept tag {tag}.")

    def reset(self):
        """Resets the dispenser."""
        self._tags = []
        self._counter = 0

    def __repr__(self):
        _next = self._tags[0] if len(self._tags) > 0 else self._counter
        return f"TagDispenser(returned={self._tags}, next={_next})"
