from collections.abc import Mapping, Hashable


class FrozenDict(Mapping, Hashable):
    """frozen dict"""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if not isinstance(k, Hashable):
                raise TypeError(f"{k} is not hashable")
            if not isinstance(v, Hashable):
                raise TypeError(f"{v} is not hashable")

        self.__dict = dict(**kwargs)
        self.__hash = None

    def __iter__(self):
        return iter(self.__dict)

    def __len__(self):
        return len(self.__dict)

    def __getitem__(self, key):
        return self.__dict[key]

    def __hash__(self):
        if self.__hash is None:
            hash_ = 0
            for pair in self.items():
                hash_ ^= hash(pair)
            self.__hash = hash_
        return self.__hash
