import numpy as np
from scipy import signal

from ..distributions.api import Normal
from .module import Module


def get_zi(b, x):
    # This is the approach NARF uses. If the initial value of x[0] is 1,
    # this is identical to the NEMS approach. We need to provide zi to
    # lfilter to force it to return the final coefficients of the dummy
    # filter operation.
    n_taps = len(b)
    null_data = np.full(n_taps*2, x[0])
    zi = np.ones(n_taps-1)
    return signal.lfilter(b, [1], null_data, zi=zi)[1]


def fir_filter(x, coefficients):
    result = []
    for x, c in zip(x, coefficients):
        zi = get_zi(c, x)
        r, zf = signal.lfilter(c, [1], x, zi=zi)
        result.append(r[np.newaxis])
    result = np.concatenate(result)
    return np.sum(result, axis=-2, keepdims=True)


def theano_convolution_node(x, coefficients):
    import theano
    from theano.tensor.signal.conv import conv2d
    theano.config.compute_test_value = 'ignore'

    def conv1d(a, b):
        a = a.dimshuffle(['x', 0])
        b = b.dimshuffle(['x', 0])
        result = conv2d(a, b, border_mode='full')[0]
        return result

    output, updates = theano.scan(conv1d, sequences=[x, coefficients])
    return output.sum(axis=0)
    # to make the convolution a function and test it
    #conv_rows = theano.function(inputs=[signal, coefficients],  outputs=final_output,
    #                            updates=updates)


    #v1_value = np.arange((12)).reshape((2, 6)).astype(theano.config.floatX)
    #c1_value = np.arange((4)).reshape((2, 2)).astype(theano.config.floatX)

    #conv_rows(v1_value, c1_value)


class FIR(Module):

    def __init__(self, n_taps, input_name='pred', output_name='pred'):
        self.n_taps = n_taps
        self.input_name = input_name
        self.output_name = output_name

    def evaluate(self, data, phi):
        coefficients = phi['coefficients']
        x = data[self.input_name]
        return {
            self.output_name: fir_filter(x, coefficients)
        }

    def generate_tensor(self, data, phi):
        coefficients = phi['coefficients']
        x = data[self.input_name]
        output = theano_convolution_node(x, coefficients)
        discard = self.n_taps-1
        output = output[discard:]

        return {
            self.output_name: output,
        }

    def get_priors(self, initial_data):
        x = initial_data[self.input_name]
        n_inputs = x.shape[0]
        prior_shape = n_inputs, self.n_taps
        c_mu = np.full(prior_shape, 1/self.n_taps, dtype='float32')
        c_sd = np.ones(prior_shape, dtype='float32')
        return {
            'coefficients': Normal(mu=c_mu, sd=c_sd),
        }
