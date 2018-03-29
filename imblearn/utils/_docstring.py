class Substitution(object):
    """Decorate a function's or a class' docstring to perform string
    substitution on it.

    This decorator should be robust even if obj.__doc__ is None
    (for example, if -OO was passed to the interpreter)
    """

    def __init__(self, *args, **kwargs):
        if (args and kwargs):
            raise AssertionError("Only positional or keyword args are allowed")

        self.params = args or kwargs

    def __call__(self, obj):
        obj.__doc__ = obj.__doc__.format(**self.params)
        return obj

    def update(self, *args, **kwargs):
        """
        Update self.params with supplied args.

        If called, we assume self.params is a dict.
        """

        self.params.update(*args, **kwargs)

    @classmethod
    def from_params(cls, params):
        """
        In the case where the params is a mutable sequence (list or dictionary)
        and it may change before this class is called, one may explicitly use a
        reference to the params rather than using *args or **kwargs which will
        copy the values and not reference them.
        """
        result = cls()
        result.params = params
        return result
