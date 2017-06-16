import numpy as np
import importlib as il
 
def fir15(obj):
    obj.queue=obj.queue+[FIR] #FIR does not need to know how many parameters it has!
    parms=create_FIR(obj.dims,15)
    obj.param_queue=obj.param_queue+[parms]
