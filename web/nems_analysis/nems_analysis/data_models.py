""" define data models here, to be imported by the data initializer """
""" TODO: would this actually be useful? organizes data into objects relationships
    instead of tables, but abstracts away from the database structure and adds
    a lot of extra stuff - leaving as dataframes for now, may come back to this"""

class Analysis():

    def __init__(self):
        pass
    
class Batch():
    
    def __init__(self):
        pass
    
    
class Cell():
    def __init__(self):
        pass
    
    
class Result():
    def __init__(self):
        pass


# build series of objects by assigning attributes based on dataframe data
# would be more OO-friendly but doesn't seem like it would add much
def build_analyses(dataframe):
    pass

def build_batches(dataframe):
    pass

def build_cells(dataframe):
    pass

def build_results(dataframe):
    pass