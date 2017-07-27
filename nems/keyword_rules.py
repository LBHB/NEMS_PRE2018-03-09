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

import sys
import inspect

def keyword_test_routine(modelname):
    """ Runs the check_keywords() method of each test class defined in this
        module (except for the Keyword_Test base class) and throws the error
        defined in the related class if a test fails.
        
    """
    
    clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    classes = [c[1] for c in clsmembers if c[1].__module__ == __name__]
    for test_class in classes:
        tester = test_class()
        passed = tester.check_keywords(modelname)
        if not passed:
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
        else:
            return False
        