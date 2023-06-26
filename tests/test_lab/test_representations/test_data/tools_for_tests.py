from hypothesis.strategies import from_type

def everything_except(excluded_types):
    return (
        from_type(type)
        .flatmap(from_type)
        .filter(lambda x: not isinstance(x, excluded_types))
    )


def factory(cls, *args, **kwargs):
    r""" Factory method which generates an instance of cls parameterized by the given arguments."""
    return cls(*args, **kwargs)

