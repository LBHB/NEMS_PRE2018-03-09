from .module import Module


class Sum(Module):

    def __init__(self, input_name, output_name):
        self.input_name = input_name
        self.output_name = output_name

    def evaluate(self, data, phi):
        x = data[self.input_name]
        return {
            self.output_name: x.sum(axis=0, keepdims=True)
        }
