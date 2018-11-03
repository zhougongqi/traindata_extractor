class Vividict(dict):
    def __missing__(self, key):
        # make multi-key assignment possible
        value = self[key] = type(self)()
        return value

    def walk(self):
        # flattened dict output
        for key, value in self.items():
            if isinstance(value, Vividict):
                for tup in value.walk():
                    yield (key,) + tup
            else:
                yield key, value
