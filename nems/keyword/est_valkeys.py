import re
import nems.modules as nm
from .registry import keyword_registry


def ev(stack, fraction):
    """
    Breaks the data into estimation and validation datasets based on the number
    of trials of each stimulus.
    """
    stack.append(nm.est_val.standard, valfrac=fraction)


def xval(stack, fraction):
    '''
    Splits the data into estimation and validation datasets by placing
    (1-fraction)*100% the trials/stimuli into the estimation set and
    fraction*100% into the validation set.
    '''
    stack.append(nm.est_val.crossval, valfrac=fraction)


def parse_ev(groups):
    fraction = float(groups[0])/100.0
    return partial(ev, fraction=fraction)


def parse_xval(groups):
    fraction = float(groups[0])/100.0
    return partial(xval, fraction=fraction)


keyword_registry.update({
    re.compile(r'xval(\d{2})'): parse_xval,
    re.compile(r'ev(\d{2})'): parse_ev,
})
