import re
import nems.modules as nm


def ev(stack, fraction):
    """
    Breaks the data into estimation and validation datasets based on the number
    of trials of each stimulus.
    """
    module = nm.est_val.standard(valfrac=fraction)
    stack.append(module)


def xval(stack, fraction):
    '''
    Splits the data into estimation and validation datasets by placing
    (1-fraction)*100% the trials/stimuli into the estimation set and
    fraction*100% into the validation set.
    '''
    module = nm.est_val.crossval(valfrac=fraction)
    stack.append(module)


def parse_ev(groups):
    fraction = float(groups[0])/100.0
    return partial(ev, fraction=fraction)


def parse_xval(groups):
    fraction = float(groups[0])/100.0
    return partial(xval, fraction=fraction)


from .registry import keyword_registry

keyword_registry.update({
    re.compile(r'xval(\d{2})'): parse_xval,
    re.compile(r'ev(\d{2})'): parse_ev,
})
