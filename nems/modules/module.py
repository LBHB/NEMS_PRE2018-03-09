class Module:

    def evaluate(self, data, phi):
        raise NotImplementedError

    def generate_tensor(self, data, phi):
        return self.evaluate(data, phi)

    def get_priors(self, initial_data):
        return {}

    def from_json(self, json_dict):
        raise NotImplementedError

    def to_json(self):
        return {}
