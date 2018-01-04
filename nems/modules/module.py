class BaseModule:

    def evaluate(self, data, phi):
        raise NotImplementedError

    def get_priors(self, initial_data):
        raise NotImplementedError

    @property
    def output_signals(self):
        raise NotImplementedError

    @property
    def input_signals(self):
        raise NotImplementedError

    def from_json(self, json_dict):
        raise NotImplementedError

    def to_json(self):
        raise NotImplementedError
