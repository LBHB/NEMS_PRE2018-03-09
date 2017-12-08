import collections


class KeywordRegistry(collections.MutableMapping):

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        for pattern, function in self.store.items():
            try:
                # Check to see if it's a a regular expression pattern
                match = pattern.match(key)
                if match is not None:
                    return function(match.groups())
            except AttributeError:
                # If not, then check to see if it's a simple string
                if pattern == key:
                    return function
        raise KeyError

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def model_name_to_stack(self, model_name):
        stack = []
        for key in model_name.split('_'):
            self[key](stack)
        return stack


keyword_registry = KeywordRegistry()
