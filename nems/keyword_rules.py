""" List of classes for checking the validity of a set of keywords.
    If a new keyword is added that should follow some kind of rule
    (i.e. 'fir10 or fir15 must be included'), a subclass of Keyword_Test
    should be added to this module to check that the rule is followed.
    Each check_keywords method should take a modelname
    (i.e. a string of keywords of the form 'keyword_keyword_keyword_...')
    as its only argument, and return True if the rule has been followed
    or False otherwise. Each tester class should also define an error attribute
    that can be thrown by the keyword_test_routine() function.
    
    Used by nems.web.nems_analysis.views when editing or creating a new
    analysis to prevent the use of invalid keyword combinations that would
    either raise an error or crash the application.
    Also used by nems.web.model_functions.views to run the same check before
    a model is fit (just incase a bad keyword combination was introduced).
    
    IMPORTANT: Any custom exceptions defined in this module (to be used by the 
    tester classes) MUST INCLUDE 'Error' in their class name so that they
    will be excluded from the keyword_test_routine.
    
"""

### Note: Disconnected calls to this module from the web interface since the
### rules are causing issues as the stack/modules/etc change around.
### If used in the future, will need to re-think which rules are appropritate
### (or if it would be better to just enforce behavior through the modules)

import sys
import inspect
import pkgutil as pk

import nems.keyword as nk
import nems.keyword.keyhelpers as nn

def keyword_test_routine(modelname):
    """ Runs the check_keywords() method of each test class defined in this
        module and throws the error defined in the related class
        if a test fails.
        
    """
    
    clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    classes = [c[1] for c in clsmembers if c[1].__module__ == __name__]
    for test_class in classes:
        if 'Error' in test_class.__name__:
            pass
        tester = test_class()
        if not tester.check_keywords(modelname):
            raise tester.error
            

class Keyword_Test():
    
    def __init__(self):
        self.error = Exception(
                'Error message that should be displayed if the test fails'
                )
    def __repr__(self):
        return 'Base class for keyword tests'
        
    def check_keywords(self, modelname):
        # Test that should be run on the modelname to determine if a rule
        # was followed.
        if True:
            return True
        else:
            return False
        

# TODO: update this when other modules that can substitute for FIR are added
class FIR_Included(Keyword_Test):

    def __init__(self):
        self.error = Exception(
                'A keyword for a FIR module must be included in the model tree'
                '\n Failed test: %s'%self.__repr__()
                )
    def __repr__(self):
        return 'FIR_Included'
    
    def check_keywords(self, modelname):
        if 'fir' in modelname:
            return True
        elif 'perfectpupil' in modelname:
            return True
        else:
            return False
        
class Nested_At_End(Keyword_Test):
    def __init__(self):
        self.error=Exception(
                'If nested crossval is included, it must be the last keyword'
                '\n Failed test: %s' % self.__repr__()
                )
    def __repr__(self):
        return 'Nested_At_End'
    
    def check_keywords(self, modelname):
        names=modelname.split('_')
        if 'nested' in modelname:
            if 'nested' in names[-1]:
                return True
            else:
                print('Nested crossval must be last keyword in modelname string')
                return False
        else:
            return True
        
        
class Keywords_Exist(Keyword_Test):
    def __init__(self):
        self.missing_kw = 'This test should not have failed yet.'
        self.error = ''
    def __repr__(self):
        return 'Keywords_Exist'
    
    def check_keywords(self,modelname):
        # put this in at top to turn off test for now, not working
        #kwtuples = inspect.getmembers(
                #sys.modules[nk.__name__], inspect.isfunction
                #)
        #kwfuncs = [k[0] for k in kwtuples]
        keywords = modelname.split('_')
        for kw in keywords: 
            found=False
            print(kw)
            if 'nested' in kw:
                if hasattr(nn,kw) is True:
                    break
                else:
                    self.missing_kw = kw
                    self.error = Exception(
                            'Keyword does not exist: {0}   '
                            '\n Failed test: {1}'
                            .format(self.missing_kw, self.__repr__())
                            )
                    return False
            else:
            #if kw in kwfuncs:
                #continue
                for importer, modname, ispkg in pk.iter_modules(nk.__path__):
                    if hasattr(importer.find_module(modname).load_module(modname),kw) is True:
                        found=True
                if found is False:
                    self.missing_kw = kw
                    self.error = Exception(
                            'Keyword does not exist: {0}   '
                            '\n Failed test: {1}'
                            .format(self.missing_kw, self.__repr__())
                            )
                    return False
                else: 
                    continue
        return True

    
    
    
    
