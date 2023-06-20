from hypothesis.strategies import from_type

def everything_except(excluded_types):
    return (
        from_type(type)
        .flatmap(from_type)
        .filter(lambda x: not isinstance(x, excluded_types))
    )