"""
- Used to parse modelstring from NarfAnalysis
- into a list of model names that can be passed back
- to model selector
"""

import ast
import re

class ModelFinder():
    
    def __init__(self, modelstring=''):
        self.modelstring = modelstring
        # as soon as modelstring is passed, go ahead and 
        # parse into array then to list so that attribs can be retrieved
        # by nemsweb.py
        self.nestedlist = self.string_to_nested_list()
        self.comboArray = []
        self.traverse_nested(self.nestedlist, [])
        self.modellist = self.array_to_list(self.comboArray)
        
    def string_to_nested_list(self):        
        # replace curly braces with brackets and remove whitespace,
        # do other string cleanup
        # then interpret as nested list
        
        
        # TODO: May not be a complete list of fixes - have been playing
        # whack-a-mole with syntax errors so far. ast.literal_eval is very picky
        # about commas, brackets etc, but still seems more efficient than writing a
        # new function for evaluating the string.
        
        s = self.modelstring.replace('{', '[')
        s = s.replace('}', ']')
        s = s.replace(" ","")
        s = s.replace('],]',']]')   #remove trailing commas
        s = s.replace('  ',' ')     #remove extra spaces
        #insert comma between adjacent quotation marks unless it's an empty
        #string
        r = re.compile(r"(?P<ONE>\w)''(?P<TWO>\w)")
        s = r.sub("\g<ONE>','\g<TWO>", s)
        #insert comma between empty string and regular string (empty first)
        r = re.compile(r"(?P<ONE>'')(?P<TWO>'\w)")
        s = r.sub("\g<ONE>,\g<TWO>", s)
        #insert comma between empty string and regular string (empty second)
        r = re.compile(r"(?P<ONE>'\w)(?P<TWO>'')")
        s = r.sub("\g<ONE>,\g<TWO>", s)
        s = s.replace("]'","],'")   #insert commas between lists & strings
        s = s.replace("][","],[")   #insert commas between lists
        
        
        try:
            nestedlist = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            print("\n\n ast.literal_eval has issue with string format for this modeltree: \n\n"\
                  + self.modelstring + "\n\n")
            return (['ast','eval','did','not','work','for','this','analysis'])
            
        return nestedlist
        
    def traverse_nested(self,nestedlist, path):
        # if passed empty list, must have reached 'leaf'
        if len(nestedlist) == 0:
            self.comboArray.append(path)

        elif type(nestedlist) is list:
            head = nestedlist[0]
            tail = nestedlist[1:]
            # if first element is a string, just append and move on
            if type(head) is str:
                self.traverse_nested(tail, (path+[head]) )
            # if it's a list, iteratively append each item to a new path
            # and traverse the rest of the list on each path
            elif type(head) is list:
                for l in range(len(head)):
                    if type(head[l]) is list:
                        self.traverse_nested(tail,path+head[l])
                    elif type(head[l]) is str:
                        self.traverse_nested(tail,path+[head[l]])
        else:
            # some kind of error message, i.e. not a list
            pass
        
        return
    
    def array_to_list(self,array):
        #ignore blank strings when joining underscores
        models = ['_'.join(filter(None,c)) for c in self.comboArray]
        
        return models