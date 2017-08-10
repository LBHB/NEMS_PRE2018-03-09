""" Helper functions for filtering cells and forming data matrix. """

from nems.db import NarfBatches

def filter_cells(batch, session, cells, min_snr=0, min_iso=0, min_snri=0):
    """ Returns a list of cells that don't meet the minimum snr/iso/snri
    criteria specified. The calling function can then remove them from the
    cell list if desired (ex: for cell in bad_cells, cell_list.remove(cell))
    
    Arguments:
    ----------
    batch : int
        The batch number to query.
    cells : list
        A list of the cellids to be checked.
    min_snr : float
        The minimum signal to noise ratio desired.
    min_iso : float
        The minimum isolation value desired.
    min_snri : float
        The minimum signal to noise ratio index desired.
    session : object
        An open database session object for querying NarfBatches.
        
    """
    
    bad_cells=[]
    
    for cellid in cells:
        dbCriteria = (
                session.query(NarfBatches)
                .filter(NarfBatches.batch == batch)
                .filter(NarfBatches.cellid.ilike(cellid))
                .first()
                )
        if dbCriteria:
            
            db_snr = min(dbCriteria.est_snr, dbCriteria.val_snr)
            db_iso = dbCriteria.min_isolation
            db_snri = dbCriteria.min_snr_index
            
            a = (min_snr > db_snr)
            b = (min_iso > db_iso)
            c = (min_snri > db_snri)
            
            if a or b or c:
                bad_cells.append(cellid)
                
                # Uncomment section below to include verbose output of
                # why individual cells were 'bad'
                
                #filterReason = ""
                #if a:
                #    filterReason += (
                #            "min snr: %s -- was less than criteria: %s\n"
                #            %(db_snr, min_snr)
                #            )
                #if b:
                #    filterReason += (
                #            "min iso: %s -- was less than criteria: %s\n"
                #            %(db_iso, min_iso)
                #            )
                #if c:
                #    filterReason += (
                #            "min snr index: %s -- was less than criteria: %s\n"
                #            %(db_snri, min_snri)
                #            )
                #print(
                #    "Removing cellid: %s,\n"
                #    "because: %s"
                #    %(cellid, filterReason)
                #    )
        else:
            print(
                "No entry in NarfBatches for cellid: {0} in batch: {1}"
                .format(cellid, batch)
                )
            bad_cells.append(cellid)
            
    print("Number of bad cells to snr/iso criteria: {0}".format(len(bad_cells)))
    print("Out of total cell count: {0}".format(len(cells)))
    
    return bad_cells