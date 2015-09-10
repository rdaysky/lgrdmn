__all__ = ["EnumExtra"]

import itertools

try:
    import enum
except:
    import enum34 as enum

class EnumExtraMeta(enum.EnumMeta):
    def __new__(cls, name, bases, classdict):
        extra_names = classdict.pop("_eex_names_", None)
        if extra_names is None:
            for b in bases:
                extra_names = type(b).__getattribute__(b, "_eex_names_")
                if extra_names is not None:
                    break

        res = enum.EnumMeta.__new__(cls, name, bases, classdict)
        res._eex_names_ = extra_names

        res._eex_value2member_map_ = { en: {} for en in extra_names }

        for member in res._member_map_.values():
            for en, ev in zip(extra_names, member.value_tuple[1:]):
                try:
                    res._eex_value2member_map_[en][ev] = member
                except TypeError:
                    pass

        return res

    def __call__(_cls, *a, **k):
        if _cls is EnumExtra:
            assert not k
            return EnumExtraMeta("EnumExtra", (EnumExtra,), { "_eex_names_": a })

        if not k:
            return super(EnumExtraMeta, _cls).__call__(*a)

        if len(k) == 1:
            (en, ev), = k.items()

            try:
                lookup = _cls._eex_value2member_map_[en]
            except:
                raise ValueError("'%s' is not a valid %s extra name" % (en, _cls.__name__))

            try:
                if ev in lookup:
                    return lookup[ev]
            except TypeError:
                # O(n)
                ix = _cls._eex_names_.index(en) + 1
                for member in _cls._member_map_.values():
                    if member.value_tuple[ix] == ev:
                        return member
            raise ValueError("no %s found with %s=%r" % (_cls.__name__, en, ev))

        raise TypeError("must provide at most one keyword argument")

def _eex_new(cls, *value_tuple):
    obj = object.__new__(cls)
    obj._value_ = value_tuple[0]
    if len(value_tuple) - 1 > len(cls._eex_names_):
        raise TypeError("too many extra values")
    for en, ev in itertools.izip_longest(cls._eex_names_, value_tuple[1:]):
        setattr(obj, en, ev)
    obj.value_tuple = value_tuple
    return obj

EnumExtra = EnumExtraMeta("EnumExtra", (enum.Enum,), { "_eex_names_": (), "__new__": _eex_new })
