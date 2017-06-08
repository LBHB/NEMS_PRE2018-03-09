from flask import render_template, jsonify, request
from nems_analysis import app, Session, NarfAnalysis, NarfBatches, NarfResults
from nems_analysis.ModelFinder import ModelFinder
import pandas.io.sql as psql
from sqlalchemy.orm import Query
from sqlalchemy import desc, asc
from flask import Response

# TODO: Figure out how to use SQLAlchemy's built-in flask context support
#       to avoid having to manually open and close a db session for each
#       request context.
# NOTE: Also need more long-term testing to see if this fixes the database
#       error issues with remote hosting.


##################################################################
####################   UI UPDATE FUNCTIONS  ######################
##################################################################


# landing page
@app.route('/')
def main_view():
    session = Session()
    
    # .all() returns a list of tuples, list comprehension to pull tuple
    # elements out into list
    analysislist = [i[0] for i in session.query(NarfAnalysis.name).order_by\
                    (desc(NarfAnalysis.lastmod)).all()]
    batchlist = [i[0] for i in session.query(NarfAnalysis.batch).distinct().all()]
    
    ######  DEFAULT SETTINGS FOR RESULTS DISPLAY  ################
    # TODO: let user choose their defaults and save for later sessions
    #cols are in addition to cellid, modelname and batch which are set up to be required
    defaultcols = ['r_test','r_fit','n_parms','batch']
    defaultrowlimit = 500
    defaultsort = 'cellid'
    ##############################################################
    
    # hard coded for now, but maybe a better way to get these
    # can't simply pull cols from NarfResults b/c includes other cols as well.
    # could remove them afterward (like cellid) but then have to add removals if
    # other cols added in the future, which defeats the point of pulling them
    # dynamically from database
    measurelist = ['r_ceiling','r_test','r_fit','r_active','mse_test','mse_fit',\
                   'mi_test','mi_fit','nlogl_test','nlogl_fit','cohere_test',\
                   'cohere_fit']
    
    statuslist = [i[0] for i in session.query(NarfAnalysis.status).distinct().all()]
    tags = [i[0].split(",") for i in session.query(NarfAnalysis.tags).distinct().all()]
    taglistdups = [i for sublist in tags for i in sublist]
    taglist = list(set(taglistdups))
    taglist.sort()
    
    # returns all columns in the format 'NarfResults.columnName'
    # then removes the leading 'NarfResults.' from each string
    collist = ['%s'%(s) for s in NarfResults.__table__.columns]
    collist = [s.replace('NarfResults.','') for s in collist]
    
    # remove cellid and modelname from options, make them required
    collist.remove('cellid')
    collist.remove('modelname')

    session.close()
    
    return render_template('main.html',analysislist = analysislist,\
                           batchlist = batchlist, collist=collist,\
                           defaultcols = defaultcols,measurelist=measurelist,\
                           defaultrowlimit = defaultrowlimit,sortlist=collist,\
                           defaultsort=defaultsort,statuslist=statuslist,\
                           taglist=taglist\
                           )
    
@app.route('/update_batch')
def update_batch():
    session = Session()
    
    aSelected = request.args.get('aSelected',type=str)
    
    batch = session.query(NarfAnalysis.batch).filter\
            (NarfAnalysis.name == aSelected).first()[0]
    
    session.close()
    
    return jsonify(batch = batch)
    

@app.route('/update_models')
def update_models():
    session = Session()
    
    aSelected = request.args.get('aSelected',type=str)
    
    modeltree = session.query(NarfAnalysis.modeltree).filter\
                (NarfAnalysis.name == aSelected).first()[0]
                
    mf = ModelFinder(modeltree)
    modellist = mf.modellist
    
    session.close()
    
    return jsonify(modellist=modellist)

@app.route('/update_cells')
def update_cells():
    session = Session()
    #just use first 3 indices of str to get the integer-only representation
    bSelected = request.args.get('bSelected',type=str)[:3]

    celllist = [i[0] for i in session.query(NarfBatches.cellid).filter\
               (NarfBatches.batch == bSelected).all()]
               
    session.close()
    
    return jsonify(celllist=celllist)

@app.route('/update_results')
def update_results():
    session = Session()
    
    nullselection = 'MUST SELECT A BATCH AND ONE OR MORE CELLS AND ONE OR MORE\
                    MODELS BEFORE RESULTS WILL UPDATE'
    
    bSelected = request.args.get('bSelected')
    cSelected = request.args.getlist('cSelected[]')
    mSelected = request.args.getlist('mSelected[]')
    colSelected = request.args.getlist('colSelected[]')
    if (len(bSelected) == 0) or (len(cSelected) == 0) or (len(mSelected) == 0):
        return jsonify(resultstable=nullselection)
    bSelected = bSelected[:3]
    
    #always add cellid, batch and modelname to column lists - required for
    #selection behavior.
    cols = [getattr(NarfResults,'cellid'),getattr(NarfResults,'modelname')]
    cols += [getattr(NarfResults,c) for c in colSelected if hasattr(NarfResults,c)]
    
    rowlimit = request.args.get('rowLimit',500)
    ordSelected = request.args.get('ordSelected')
    if ordSelected == 'asc':
        ordSelected = asc
    elif ordSelected == 'desc':
        ordSelected = desc
    sortSelected = request.args.get('sortSelected','cellid')


    results = psql.read_sql_query(Query(cols,session).filter\
              (NarfResults.batch == bSelected).filter\
              (NarfResults.cellid.in_(cSelected)).filter\
              (NarfResults.modelname.in_(mSelected)).order_by(ordSelected\
              (getattr(NarfResults,sortSelected))).limit(rowlimit).statement,\
              session.bind)
    
    session.close()
    
    return jsonify(resultstable=results.to_html(classes='table-hover\
                                                table-condensed'))


@app.route('/update_analysis')
def update_analysis():
    session = Session()
    
    tSelected = request.args.get('tSelected')
    sSelected = request.args.get('sSelected')
    
    if tSelected == '__any':
        tString = '%'
    else:
        tString = '%' + tSelected + '%'
    
    if sSelected == '__any':
        sString = '%'
    else:
        sString = sSelected
    
    analysislist = psql.read_sql_query(session.query(NarfAnalysis).filter\
                                   (NarfAnalysis.tags.ilike(tString)).filter\
                                   (NarfAnalysis.status.ilike(sString)).statement,\
                                   session.bind)
    
    session.close()
    
    return jsonify(analysislist = analysislist)

@app.route('/update_analysis_details')
def update_analysis_details():
    session = Session()
    # additional columns to display in detail popup - add/subtract here if desired.
    detailcols = ['id','status','question','answer']
    
    aSelected = request.args.get('aSelected')
    
    cols = [getattr(NarfAnalysis,c) for c in detailcols if hasattr(NarfAnalysis,c)]
    
    results = psql.read_sql_query(Query(cols,session).filter\
                                  (NarfAnalysis.name == aSelected).statement,session.bind)
    
    detailsHTML = """"""
    
    for col in detailcols:
        #single line for id or status
        if (col == 'id') or (col == 'status'):
            detailsHTML += """<p><strong>%s</strong>: %s</p>
                           """%(col,results.get_value(0,col))
        #header + paragraph for anything else
        else:
            detailsHTML += """<h5><strong>%s</strong>:</h5>
                          <p>%s</p>
                          """%(col,results.get_value(0,col))
                    
    session.close()
    
    return jsonify(details=detailsHTML)


####################################################################
###############     TABLE SELECTION FUNCTIONS     ##################
####################################################################

@app.route('/get_preview')
def get_preview():
    session = Session()
    
    bSelected = request.args.get('bSelected',type=str)[:3]
    cSelected = request.args.getlist('cSelected[]')
    mSelected = request.args.getlist('mSelected[]')

    paths = psql.read_sql_query(session.query(NarfResults.figurefile).filter\
                               (NarfResults.batch == bSelected).filter\
                               (NarfResults.cellid.in_(cSelected)).filter\
                               (NarfResults.modelname.in_(mSelected)).statement,\
                               session.bind)
    
    filepaths = paths['figurefile'].tolist()
    
    if len(filepaths) == 0:
        return jsonify(filepaths=['/missing_preview'])
    return jsonify(filepaths=filepaths)
        
@app.route('/missing_preview')
def missing_preview():
    return Response('No preview image exists for this result')

@app.route('/preview/<path:filepath>')
def preview(filepath):

    try:
        with open('/' + filepath, 'r+b') as img:
            image = img.read()

        return Response(image,mimetype="image/png")
    except:
        return Response('Image path exists in DB but image not in local storage')


####################################################################
###################     MISCELLANEOUS  #############################
####################################################################


# clicking error log link will open text file with notes
# TODO: add interface to edit text from site, or submit notes some other way,
# so that users can report bugs/undesired behavior
@app.route('/error_log')
def error_log():
    return app.send_static_file('error_log.txt')
