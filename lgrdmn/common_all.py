# File encoding: UTF-8

import inspect
import itertools
import json
import re
import six
import types
import csv

def namespace(function):
    fields = function()
    assert isinstance(fields, dict), "Function to be used as namespace must return a dict, usually by `return locals()`"

    fields = { k: (staticmethod(v) if isinstance(v, types.FunctionType) else v)
        for k, v in fields.items() if not k.startswith("_") }

    fields["__doc__"] = function.__doc__

    ns = type(function.func_name, (), fields)
    ns.__module__ = function.__module__

    return ns

def force_unicode(s, encoding="UTF-8", errors="strict"): # taken from django.utils.encoding.force_text
    if isinstance(s, six.text_type):
        return s

    if isinstance(s, six.string_types):
        return s.decode(encoding, errors)

    if six.PY3:
        return six.text_type(s, encoding, errors) if isinstance(s, bytes) else six.text_type(s)

    if hasattr(s, "__unicode__"):
        return six.text_type(s)

    return six.text_type(bytes(s), encoding, errors)

def utf8(x):
    if isinstance(x, unicode):
        return x.encode("UTF-8")
    return x

def force_utf8(x):
    if isinstance(x, str):
        return x
    return unicode(x).encode("UTF-8")

def recursive_map(f, x):
    if isinstance(x, list):
        return [recursive_map(f, i) for i in x]
    if isinstance(x, tuple):
        return tuple(recursive_map(f, i) for i in x)
    if isinstance(x, dict):
        return { recursive_map(f, k): recursive_map(f, v) for k, v in x.items() }
    return f(x)

class csv_writer_utf8:
    def __init__(self, *a, **k):
        self.writer = csv.writer(*a, **k)

    def writerow(self, row):
        return self.writer.writerow(recursive_map(utf8, row))

    def writerows(self, rows):
        return self.writer.writerows(recursive_map(utf8, rows))

    @property
    def dialect(self):
        return self.writer.dialect

_re_whitespace = re.compile(r"\s+")
def normalize_space(s):
    return re.sub(_re_whitespace, " ", s).strip()

def groupby(objects, func):
    return [(i, list(j)) for i, j in itertools.groupby(objects, func)]

def argument_bindings(func, args, kwargs):
    # http://code.activestate.com/recipes/551779-introspecting-call-arguments/
    """
    Get the actual value bound to each formal parameter when calling `func(*args, **kwargs)`.

    It works for methods too (bounded, unbounded, staticmethods, classmethods).

    @returns: `(bindings, missing_args)`, where:
        - `bindings` is a mapping of every formal parameter (including *args
           and **kwargs if present) of the function to the respective bound value.
        - `missing_args` is a tuple of the formal parameters whose value was not
           provided (i.e. using the respective default value)

    Raises TypeError if the function is not callable with the given arguments.

    Examples::
        >>> class X:
        ...     def m(self, a, b, c=1, d=2, *args, **kwargs):
        ...         pass
        ...     @classmethod
        ...     def c(cls, a, b, c=1, d=2, *args, **kwargs):
        ...         pass
        ...     @staticmethod
        ...     def s(a, b, c=1, d=2, *args, **kwargs):
        ...         pass
        ...     def __repr__(self):
        ...         return "<X>"
        >>> def f(a, b, c=1, d=2, *args, **kwargs):
        ...     pass
        >>> def k(a, b, c=1, d=2, **kwargs):
        ...     pass
        >>> x = X()
        >>> argument_bindings(X.m, x, 10, 20, d=30, e=40)
        ({'a': 10, 'c': 1, 'b': 20, 'd': 30, 'self': <X>, 'args': (), 'kwargs': {'e': 40}}, ('c',))
        >>> argument_bindings(X.m, 10, 20, 30)
        Traceback (most recent call last):
            bada boom
        TypeError: unbound method m() must be called with X instance as first argument (got int instance instead)
        >>> argument_bindings(X.m, b=20, a=10)
        Traceback (most recent call last):
            bada boom
        TypeError: unbound method m() must be called with X instance as first argument (got nothing instead)
        >>> argument_bindings(x.m, b=20, a=10)
        ({'a': 10, 'c': 1, 'b': 20, 'd': 2, 'self': <X>, 'args': (), 'kwargs': {}}, ('c', 'd'))
        >>> argument_bindings(X.s, 10, 20, 30)
        ({'a': 10, 'c': 30, 'b': 20, 'd': 2, 'args': (), 'kwargs': {}}, ('d',))
        >>> argument_bindings(f, 10, 20, 30, 40, 50)
        ({'a': 10, 'c': 30, 'b': 20, 'd': 40, 'args': (50,), 'kwargs': {}}, ())
        >>> argument_bindings(k, 10, 20, 30, 40, 50)
        Traceback (most recent call last):
            bada boom
        TypeError: k() takes at most 4 arguments (5 given)
    """

    # syntax highlighing goes crazy here for some reason, so this is needed -> """

    # the following code mutates these two variables
    args = list(args)
    kwargs = dict(kwargs)

    spec_args, spec_varargs, spec_keywords, spec_defaults = inspect.getargspec(func)
    bindings = {}

    if inspect.ismethod(func):
        # implicit 'self' (or 'cls' for classmethods) argument: func.im_self if present
        if func.im_self is not None:
            bindings[spec_args.pop(0)] = func.im_self
        elif not args or not isinstance(args[0], func.im_class):
            raise TypeError("unbound method %s() must be called with %s instance as first argument (got %s instead)" % (
                func.func_name,
                func.im_class.__name__,
                args and ("%s instance" % type(args[0]).__name__) if args else "nothing",
            ))

    num_args      = len(args)
    num_spec_args = len(spec_args)
    num_defaults  = len(spec_defaults or ())

    has_kwargs = bool(kwargs)

    # get the expected arguments passed positionally
    bindings.update(zip(spec_args, args))

    # get the expected arguments passed by name
    for arg in spec_args:
        if arg in kwargs:
            if arg in bindings:
                raise TypeError("%s() got multiple values for keyword argument '%s'" % (
                    func.func_name,
                    arg,
                ))
            bindings[arg] = kwargs.pop(arg)

    # fill in any missing values with the defaults
    missing = []
    if spec_defaults:
        for arg, value in zip(spec_args[-num_defaults:], spec_defaults):
            if arg not in bindings:
                bindings[arg] = value
                missing.append(arg)

    # ensure that all required args have a value
    for arg in spec_args:
        if arg not in bindings:
            num_required = num_spec_args - num_defaults
            raise TypeError("%s() takes at least %d %sargument%s (%d given)" % (
                func.func_name,
                num_required,
                "non-keyword " if has_kwargs else "",
                "s" if num_required > 1 else "",
                num_args,
            ))

    # handle any remaining named arguments
    if spec_keywords:
        bindings[spec_keywords] = kwargs
    elif kwargs:
        raise TypeError("%s() got an unexpected keyword argument '%s'" % (
            func.func_name,
            iter(kwargs).next(),
        ))

    # handle any remaining positional arguments
    if spec_varargs:
        if num_args > num_spec_args:
            bindings[spec_varargs] = args[-(num_args - num_spec_args):]
        else:
            bindings[spec_varargs] = ()
    elif num_spec_args < num_args:
        raise TypeError("%s() takes %s %d argument%s (%d given)" % (
            func.func_name,
            "at most" if spec_defaults else "exactly",
            num_spec_args,
            "s" if num_spec_args > 1 else "",
            num_args,
        ))
    return bindings, tuple(missing)

class OverloadSet(object):
    def __init__(self, callable):
        self.callables = [callable]

    def __call__(self, *args, **kwargs):
        match = None
        for callable in self.callables:
            try:
                argument_bindings(callable, args, kwargs)
            except TypeError:
                pass
            else:
                if match:
                    raise TypeError("multiple overloaded functions match")
                match = callable

        if not match:
            raise TypeError("no overloaded functions match")

        return match(*args, **kwargs)

    def overload(self, callable): # to be used as decorator
        self.callables.append(callable)
        return self

def overload(callable):
    return OverloadSet(callable)

def monkey_patch_replace_method(cls): # Decorator
    """ Replaces a method of cls with the decorated function, passing the original method as the first argument. """
    def complain(*a, **k):
        raise NotImplementedError("Function used for monkey patching can’t be called directly")

    def mp_outer(function):
        function_orig = getattr(cls, function.func_name)
        def mp_inner(*a, **k): # Not lambda to ease debugging.
            return function(function_orig, *a, **k)
        setattr(cls, function.func_name, mp_inner)
        return complain

    return mp_outer

def combine_dicts(*args, **kwargs):
    """ Updates {} with all dicts in *args and then updates with kwargs. """
    m = {}
    for i in args:
        m.update(i)
    m.update(kwargs)
    return m

def coalesce(*args):
    """ Returns first non-None argument. """
    for i in args:
        if i is not None:
            return i
    return None

JE_DEFAULT_DEFAULT = object()
def json_encoder_class(base=object):
    def je_inner(function):
        class JsonEncoder(base):
            def default(self, o):
                result = function(o)

                if result is not JE_DEFAULT_DEFAULT:
                    return result

                return super(JsonEncoder, self).default(o)
        return JsonEncoder
    return je_inner

def write_json(obj, buffer=None, indent=None, pretty=False, encoder_class=None):
    """ Calls json.dump/dumps, optionally prettifying the output.  """
    params = dict(
        cls = encoder_class,
        indent = indent if indent is not None else 2 if pretty else None,
        separators = (", ", ": ") if pretty else (",", ":"),
        sort_keys = pretty,
    )
    if buffer:
        json.dump(obj, buffer, **params)
        return buffer
    else:
        return json.dumps(obj, **params)

def superdomains(domain):
    """ ["f.q.d.n", "q.d.n", "d.n", "n"] """
    components = domain.split(".")
    for i in range(len(components)):
        yield ".".join(components[i:])

def key_for_host(keys, host):
    """ Returns keys[f.q.d.n | *.f.q.d.n | *.q.d.n | *.d.n | *.n | *] """
    for k in [host] + ["*.%s" % d for d in superdomains(host)] + ["*"]:
        if k in keys:
            return keys[k]
    raise KeyError(host)
