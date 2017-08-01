#!/usr/bin/env python

# This script runs nems_main.fit_single_model from the command line

import nems.main as nems
from nems.db import cluster_Session, tQueue
import sys
import os
import nems.db as nd

if __name__ == '__main__':
    
    # leftovers from some industry standard way of parsing inputs
    
    #parser = argparse.ArgumentParser(description='Generetes the topic vector and block of an author')
    #parser.add_argument('action', metavar='ACTION', type=str, nargs=1, help='action')
    #parser.add_argument('updatecount', metavar='COUNT', type=int, nargs=1, help='pubid count')
    #parser.add_argument('offset', metavar='OFFSET', type=int, nargs=1, help='pubid offset')
    #args = parser.parse_args()
    #action=parser.action[0]
    #updatecount=parser.updatecount[0]
    #offset=parser.offset[0]

    if 'QUEUEID' in os.environ:
        queueid=os.environ['QUEUEID']
        print("Starting QUEUEID={}".format(queueid))
        conn=nd.cluster_engine.connect()
        # tick off progress, job is live
        sql="UPDATE tQueue SET complete=-1,progress=progress+1 WHERE id={}".format(queueid)
        result = conn.execute(sql)
    else:
        queueid=0
        
    if len(sys.argv)<4:
        print('syntax: nems_fit_single cellid batch modelname')
        exit(-1)

    cellid=sys.argv[1]
    batch=sys.argv[2]
    modelname=sys.argv[3]
    
    print("Running fit_single_model({0},{1},{2})".format(cellid,batch,modelname))
    stack = nems.fit_single_model(cellid,batch,modelname,autoplot=False)

    print("Done with fit.")
    
    # Edit: added code to save preview image. -Jacob 7/6/2017
    path = stack.quick_plot_save(mode="png")
    print("Preview saved to: {0}".format(path))
    
    if queueid:
        # mark job complete
        # svd old-fashioned way of doing
        #sql="UPDATE tQueue SET complete=1 WHERE id={}".format(queueid)
        #result = conn.execute(sql)
        #conn.close()
   
        cluster_session = cluster_Session()
        # also filter based on note - should only be one result to match either
        # filter, but double checks to make sure there's no conflict
        note = "{0}/{1}/{2}".format(cellid, batch, modelname)
        qdata = (
                cluster_session.query(tQueue)
                .filter(tQueue.id == queueid)
                .filter(tQueue.note == note)
                .first()
                )
        if not qdata:
            # Something went wrong - either no matching id, no matching note,
            # or mismatch between id and note
            print("Invalid query result when checking for queueid & note match")
            print("/n for queueid: %s"%queueid)
        else:
            qdata.complete = 1
            cluster_session.commit()
       
        cluster_session.close()
           
       
