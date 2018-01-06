import logging
log = logging.getLogger(__name__)


from nems.stack import nems_stack
import collections


class KeywordRegistry(collections.MutableMapping):

    def __init__(self, *args, **kwargs):
        # The main reason I'm using an OrderedDict here is to enable us to
        # control the order of discovery of key functions. Right now I don't see
        # a use-case for this (other than in the unit-tests).
        self.store = collections.OrderedDict()
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
            except ValueError:
                # Keep going. We may eventually find a match in the list.
                pass
        log.info("Couldn't find key: {0}".format(key))
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
        stack = nems_stack()
        for key in model_name.split('_'):
            self[key](stack)
        return stack


keyword_registry = KeywordRegistry()
