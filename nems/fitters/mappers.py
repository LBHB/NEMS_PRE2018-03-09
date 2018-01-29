"""Defines mapping functions that return a packer and unpacker (in that order)
corresponding to the type of sigma expected by a fitter.

TOOD: Proposed naming convention is to use whatever the modelspec is being
      turned into as the name for the function, but maybe something else makes
      more sense?

TODO: might make more sense to put this in an analysis.utils module?
      since analyses are what actually use the packer/unpackers.
"""

from .util import phi_to_vector, vector_to_phi

def simple_vector():
    """Interconverts phi to or from a list of dictionaries
    to a single flattened vector.

    """

    # modelspec to vector
    def packer(modelspec):
        phi = [m['phi'] for m in modelspec]
        vec = phi_to_vector(phi)
        return vec

    # vector to modelspec
    def unpacker(vec, modelspec):
        phi_template = [m['phi'] for m in modelspec]
        phi = vector_to_phi(vec, phi_template)
        for i, p in enumerate(phi):
            modelspec[i]['phi'] = p
        return modelspec

    return packer, unpacker