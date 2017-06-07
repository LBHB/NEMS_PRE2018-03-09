"""
stand-alone version of fit_single_model to be called from command line by
model queue
"""

#import necessary model stuff (copy from fit_single_model)
#import sql alchemy tools to set up own database connection

#DO NOT IMPORT ANYTHING FROM NEMS_ANALYSIS!!! 
#App will not be running when called by model queue, so importing anything
#initialized by flask will cause an error

def q_fit_single_model():
    #copy paste code from fit_single_model here
    
    #but replace session=Session() with code for creating own database connection
    #should still be able to read URI from text file as in config.py
    
    #don't need to return any data in this version since nothing will be displayed
    #to user. but maybe return some kind of success/failure indicator to the
    #database?
    return