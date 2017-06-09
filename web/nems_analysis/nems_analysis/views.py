from flask import render_template, jsonify, request, redirect, url_for, Response
from nems_analysis import app, Session, NarfAnalysis, NarfBatches, NarfResults
from nems_analysis.ModelFinder import ModelFinder
import pandas.io.sql as psql
from sqlalchemy.orm import Query
from sqlalchemy import desc, asc
import datetime

# TODO: Figure out how to use SQLAlchemy's built-in flask context support
#       to avoid having to manually open and close a db session for each
#       request context?? Or just leave it manual for clarity?

# TODO: Currently, analysis edit/delete/etc are handled by name, which requires
#       enforcing a unique name for each analysis. Should re-work selector to include
#       id with each name so that it can be used instead, since it's the primary key.
#       But each analysis should probably have a unique name anyway, so low priority change.


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
                    (asc(NarfAnalysis.id)).all()]
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
    #separate tags into list of list of strings
    tags = [i[0].split(",") for i in session.query(NarfAnalysis.tags).distinct().all()]
    #flatten list of lists into a single list of all tag strings
    taglistbldupspc = [i for sublist in tags for i in sublist]
    #remove leading or trailing spaces
    taglistbldup = [t.strip() for t in taglistbldupspc]
    #reform list with only unique tags
    taglistbl = list(set(taglistbldup))
    #remove blank tags
    taglist = [t for t in taglistbl if t != '']
    #then sort alphabetically
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

# update current batch selection after analysis selected    
@app.route('/update_batch')
def update_batch():
    session = Session()
    
    aSelected = request.args.get('aSelected',type=str)
    
    batch = session.query(NarfAnalysis.batch).filter\
            (NarfAnalysis.name == aSelected).first()[0]
    
    session.close()
    
    return jsonify(batch = batch)
    

# update list of modelnames after analysis selected
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


# update list of cells after batch selected (cascades from analysis selection)
@app.route('/update_cells')
def update_cells():
    session = Session()
    #just use first 3 indices of str to get the integer-only representation
    bSelected = request.args.get('bSelected',type=str)[:3]

    celllist = [i[0] for i in session.query(NarfBatches.cellid).filter\
               (NarfBatches.batch == bSelected).all()]
               
    session.close()
    
    return jsonify(celllist=celllist)


# update table of results after batch, cell(s) and model(s) selected
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


# update list of analyses after tag and/or status filter(s) selected
@app.route('/update_analysis')
def update_analysis():
    session = Session()
    
    tagSelected = request.args.get('tagSelected')
    statSelected = request.args.get('statSelected')
    
    if tagSelected == '__any':
        tString = '%%'
    else:
        tString = '%' + tagSelected + '%'
    
    if statSelected == '__any':
        sString = '%%'
    else:
        sString = statSelected

    analysislist = [i[0] for i in session.query(NarfAnalysis.name).filter\
                    (NarfAnalysis.tags.ilike(tString)).filter\
                    (NarfAnalysis.status.ilike(sString)).order_by(\
                    asc(NarfAnalysis.id)).all()]
    
    session.close()
    
    return jsonify(analysislist = analysislist)


# update content of analysis details popover after analysis selected
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



##############################################################################
################      edit/delete/new  functions for Analysis Editor #########
##############################################################################



# takes input from analysis modal popupand saves submission to database
@app.route('/edit_analysis', methods=['GET','POST'])
def edit_analysis():
    session = Session()
    
    eName = request.form.get('editName')
    eStatus = request.form.get('editStatus')
    eTags = request.form.get('editTags')
    eQuestion = request.form.get('editQuestion')
    eAnswer = request.form.get('editAnswer')
    eTree = request.form.get('editTree')
    eBatch = request.form.get('editBatch')
    #TODO: add checks to require input inside form fields
    #       or allow blank so that people can erase stuff?
    
    modTime = str(datetime.datetime.now().replace(microsecond=0))
    
    #TODO: this requires that all analyses have to have a unique name.
    #       better way to do this or just enforce the rule?
    
    #find out if analysis with same name already exists. if it does, grab its
    #sql alchemy object and update with new values, so that same id is overwritten
    checkExists = session.query(NarfAnalysis).filter(NarfAnalysis.name == eName).all()
    if len(checkExists) > 1:
        session.close()
        return Response("Oops! More than one analysis with the same name already exists,\
                        something is wrong!")
    elif len(checkExists) == 1:
        a = checkExists[0]
        a.name = eName
        a.status = eStatus
        a.question = eQuestion
        a.answer = eAnswer
        a.tags = eTags
        a.batch = eBatch
        a.lastmod = modTime
        a.modeltree = eTree
        
    #if doesn't exist, add new sql alchemy object with appropriate attributes,
    #which should store in new row
    else:
        a = NarfAnalysis(\
                name=eName,status=eStatus,question=eQuestion,answer=eAnswer,\
                tags=eTags,batch=eBatch,lastmod=modTime,modeltree=eTree)
        session.add(a)
    
    #for verifying correct logging - can delete/comment these out when no longer
    #needed for testing.
    print("checking if attributes added correctly")
    print(a.name)
    print(a.question)
    print(a.answer)
    print(a.status)
    print(a.tags)
    print(a.batch)
    print(a.lastmod)
    print(a.modeltree)
    
    session.commit()
    session.close()
    
    #after handling submissions, return user to main page so that it
    #refreshes with new analysis included in list    
    return redirect(url_for('main_view'))


# populates editor form with fields for selected analysis
@app.route('/get_current_analysis')
def get_current_analysis():
    session = Session()
    
    aSelected = request.args.get('aSelected')
    
    if len(aSelected) == 0:
        return jsonify(name='',status='',tags='',batch='',question='',answer='',\
                       tree='')
        
    a = session.query(NarfAnalysis).filter(NarfAnalysis.name == aSelected).first()
    
    session.close()
    
    return jsonify(name=a.name,status=a.status,tags=a.tags,batch=a.batch,\
                   question=a.question,answer=a.answer,tree=a.modeltree)
        
    
# checks for duplicate analysis name on form submission,
# triggers warning via JS if analysis with same name already exists
@app.route('/check_analysis_exists')
def check_analysis_exists():
    session = Session()
    
    nameEntered = request.args.get('nameEntered')
    
    result = session.query(NarfAnalysis).filter(NarfAnalysis.name == nameEntered).first()
    
    exists = True
    if result is None:
        exists = False
        
    session.close()
    
    return jsonify(exists=exists)


# deletes selected analysis from DB
@app.route('/delete_analysis')
def delete_analysis():
    session = Session()
    
    success = True
    aSelected = request.args.get('aSelected')

    if len(aSelected) == 0:
        success = False
        return jsonify(success=success)
    
    result = session.query(NarfAnalysis).filter(NarfAnalysis.name == aSelected).first()
    
    if result is None:
        success = False
        return jsonify(success=success)

    #leaving these here incase accidental deletion or some other issue occurs.
    #that way can be copy pasted back into new analysis form to restore
    print("checking for correct deletion. Deleting:")
    print(result.name)
    print(result.tags)
    print(result.status)
    print(result.batch)
    print(result.question)
    print(result.answer)
    print(result.modeltree)
    print(result.summaryfig)

    session.delete(result)
    session.commit()
    session.close()

    return jsonify(success=success)



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
