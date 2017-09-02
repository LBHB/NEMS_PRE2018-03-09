#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 18:35:22 2017

@author: shofer
"""

"""
Initialization file for user_def_mods. 

nems_modules subclasses defined by individual users should be kept in this file.

Modules can either be imported from this package individually into nems.modules
like:
    import nems.user_def_mods.modX as udmX (or whatever name)
    
Or by adding modX to the all __all__ list below and calling:
    
    from nems.user_def_mods import *

However, using the latter method to import will import everything in
the __all__ list, which could potentially crowd your namespace, and whatever
function names you import will not be recognized by the importing file. Thus,
it is not recommended. But you do you.
"""
import nems.modules.user_def.load_baphy_ssa
import nems.modules.user_def.demo


__all__=['load_baphy_ssa','simple_demo']