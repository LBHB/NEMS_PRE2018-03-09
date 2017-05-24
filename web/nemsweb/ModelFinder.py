"""
- Used to parse modelstring from NarfAnalysis
- into a list of model names that can be passed back
- to model selector
"""

import ast
import pprint


""" for testing w/  actual model string
import QueryGenerator as qg
import DB_Connection as dbcon
db = dbcon.DB_Connection()
dbc = db.connection

analysis = qg.QueryGenerator(dbc, column='name',table='NarfAnalysis', analysis='Fitters').send_query()
globmodelstring = analysis['modeltree'][0]
"""

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
        # then interpret as nested list
        s = self.modelstring.replace('{', '[')
        s = s.replace('}', ']')
        s = s.replace(" ","")
        nestedlist = ast.literal_eval(s)
        
        return nestedlist
        
    def traverse_nested(self,nestedlist, path):
        # TODO: parse model string into an array of possible combinations
        # see: narf_analysis > keyword_combos
        
        # TODO: it's definitely building an array, but it's much too large -
        # something not quite right. reaching recursion limit.
        
        # if passed empty list, must have reached 'leaf'
        if len(nestedlist) == 0:
            self.comboArray += path
        # double check that a list was passed - otherwise we've got problems
        elif type(nestedlist) is list:
            head = nestedlist[0]
            tail = nestedlist[1:]
            # if length is 1, it's just one string (but still read as a list)
            # so concat contents onto path, then traverse the next part of list
            if len(head) == 1:
                path.append(head[0])
                self.traverse_nested(tail, path)
            # if length is greater than 1, it's a list, so
            # for each item concat to path then traverse the rest of the list
            elif len(head) > 1:
                for l in range(len(head)):
                    path.append(head[l])
                    self.traverse_nested(tail, path)
        else:
            # some kind of error message, i.e. not a list
            pass
    
    def array_to_list(self,array):
        # parse array into a list of model names (strings)
        # see: narf_analysis > rebuild_modeltree
        models = []
        
        # iterate through rows of combinations
        for c in range(len(self.comboArray)):
            model = ''
            for s in range(len(self.comboArray[c])):
                model += ("%s_" + self.comboArray[c][s])
            models += model
        
        return models