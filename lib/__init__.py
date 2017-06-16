#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12

author: @svd
"""
##Generic utitilies for handling LBHB data

#Port basic routines from baphy (Matlab) that allow use of baphy data in python:
    #   1. dbopen -- connect to database (take out of web interface?)
    #   2. dbgetscellfile -- query for sCellFile entries
    #   3. request_celldb_batch - get file info for a given batch/cell
    #   4. loadspikeraster
    #   5. loadevpraster
    #   6. LoadMFile -- load contents of baphy parameter+event file
    #   7. evtimes 
    #   8. raster_plot

#Note: I created a raster plot function in the Plotting package that can take in the response raster from the baphy file-NJS