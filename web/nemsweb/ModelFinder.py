"""
- Used to parse modelstring from NarfAnalysis
- into a list of model names that can be passed back
- to model selector
"""

import ast

#for testing w/  actual model string
import QueryGenerator as qg
import DB_Connection as dbcon
db = dbcon.DB_Connection()
dbc = db.connection

analysis = qg.QueryGenerator(dbc,tablename='NarfAnalysis', analysis='Noisy Vocalizations').send_query()
modstring = analysis['modeltree'][0]


"""
# for testing with simpler nested list that (hopefully) won't crash the universe
modstring = "{'a','b',  {{'c','d'},{ 'e','f' }},    'g'}"
# list of combos should end up as:
# ['abceg','abcfg','abdeg','abdfg']
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
        # parse array into a list of model names (strings)
        # see: narf_analysis > rebuild_modeltree
        models = []
        
        # TODO: Maybe problems with this too, but could just be due to the
        # combo array not coming out right
        
        # iterate through rows of combinations
        for c in self.comboArray:
            models += ['_'.join(c)]
        return models