"""View functions for the main page of the nems_analysis web interface.

UI Upate functions handle the initial state of the landing page as well as
refreshing analysis, batch, cell and model selectors when a user makes a
selection. The general procedure for each function is:
    -Get the user's selections from an AJAX call.
    -Query the database for a list of new entries for the dependent selector,
        based on the user's selections.
    -Return a JSON-serialized version of the list of new entries.

Analysis Editor functions populate the fields of the Analysis Editor modal,
add the data in the fields to the database, and/or delete an analysis from
the database.

Table Selection functions implement callbacks for the Preview button (so far),
as well as any other buttons whose behavior depends on selections within the
results table (as opposed to the selectors).

Miscellaneous functions handle other simple UI features that don't fit in
any other category (so far, just one function to serve error_log.txt).

"""

import datetime
import sys
from base64 import b64encode

from flask import (
        render_template, jsonify, request, redirect,url_for, Response,
        )
import pandas.io.sql as psql
from sqlalchemy.orm import Query
from sqlalchemy import desc, asc

from nems_analysis import (
        app, Session, NarfAnalysis, NarfBatches, NarfResults, sBatch,
        )
from nems_analysis.ModelFinder import ModelFinder
from plot_functions.PlotGenerator import PLOT_TYPES

# TODO: Figure out how to use SQLAlchemy's built-in flask context support
#       to avoid having to manually open and close a db session for each
#       request context? Or just leave it manual for clarity?

# TODO: Currently, analysis edit/delete/etc are handled by name, 
#       which requires enforcing a unique name for each analysis. 
#       Should re-work selector to includ id with each name so that 
#       id can be used instead, since it's the primary key.
#       But each analysis should probably have a unique name anyway, 
#       so low priority change.



##################################################################
####################   UI UPDATE FUNCTIONS  ######################
##################################################################



@app.route('/')
def main_view():
    """Initialize the nems_analysis landing page.
    
    Queries the database to get lists of available analyses, batches,
    status filters, tag filters, and results columns.
    Specifies defaults for results columns, row limit and sort column.
    
    Returns:
    --------
    main.html : template
        The landing page template rendered with variables for analysislist,
        batchlist, collist, defaultcols, measurelist, defaultrowlimit,
        sortlist, defaultsort, statuslist, and taglist.
    
    """
    
    session = Session()
    
    # .all() returns a list of tuples, so it's necessary to pull the
    # name elements out into a list by themselves.
    analysislist = [
            i[0] for i in
            session.query(NarfAnalysis.name)
            .order_by(asc(NarfAnalysis.id)).all()
            ]
    
    batchids = [
            i[0] for i in
            session.query(NarfBatches.batch)
            .distinct().all()
            ]
    batchnames = []
    for i in batchids:
        name = (
                session.query(sBatch.name)
                .filter(sBatch.id == i)
                .first()
                )
        if not name:
            batchnames.append('')
        else:
            batchnames.append(name.name)
    batchlist = [
            (batch + ': ' + batchnames[i])
            for i, batch in enumerate(batchids)
            ]
    batchlist.sort()
    
    # Default settings for results display.
    # TODO: let user choose their defaults and save for later sessions
    # cols are in addition to cellid, modelname and batch,
    # which are set up to be required
    defaultcols = ['r_test', 'r_fit', 'n_parms', 'batch']
    defaultrowlimit = 500
    defaultsort = 'cellid'
    measurelist = [
            'r_ceiling', 'r_test', 'r_fit', 'r_active', 'mse_test',
            'mse_fit', 'mi_test', 'mi_fit', 'nlogl_test',
            'nlogl_fit', 'cohere_test', 'cohere_fit',
            ]
    statuslist = [
            i[0] for i in
            session.query(NarfAnalysis.status)
            .distinct().all()
            ]
    
    # Separate tags into list of lists of strings.
    tags = [
            i[0].split(",") for i in
            session.query(NarfAnalysis.tags)
            .distinct().all()
            ]
    # Flatten list of lists into a single list of all tag strings
    # and remove leading and trailing whitespace.
    taglistbldupspc = [i for sublist in tags for i in sublist]
    taglistbldup = [t.strip() for t in taglistbldupspc]
    # Reform the list with only unique tags
    taglistbl = list(set(taglistbldup))
    # Finally, remove any blank tags and sort the list.
    taglist = [t for t in taglistbl if t != '']
    taglist.sort()
    
    # Returns all columns in the format 'NarfResults.columnName,'
    # then removes the leading 'NarfResults.' from each string
    collist = ['%s'%(s) for s in NarfResults.__table__.columns]
    collist = [s.replace('NarfResults.', '') for s in collist]
    # Remove cellid and modelname from options toggles- make them required.
    collist.remove('cellid')
    collist.remove('modelname')

    plotTypeList = PLOT_TYPES

    session.close()
    
    return render_template(
            'main.html', analysislist=analysislist, batchlist=batchlist,
            collist=collist, defaultcols=defaultcols, measurelist=measurelist,
            defaultrowlimit=defaultrowlimit,sortlist=collist,
            defaultsort=defaultsort,statuslist=statuslist, taglist=taglist,
            plotTypeList=plotTypeList,
            )


@app.route('/update_batch')
def update_batch():
    """Update current batch selection after an analysis is selected."""
    session = Session()
    
    aSelected = request.args.get('aSelected', type=str)
    
    batch = (
            session.query(NarfAnalysis.batch)
            .filter(NarfAnalysis.name == aSelected)
            .first()
            )
    try:
        batch = batch.batch
    except Exception as e:
        print(e)
        batch = ''
    
    session.close()
    
    return jsonify(batch=batch)
    

@app.route('/update_models')
def update_models():
    """Update the list of modelnames in the model selector after an
    analysis is selected.
    
    """
    
    session = Session()
    
    aSelected = request.args.get('aSelected', type=str)
    
    modeltree = (
            session.query(NarfAnalysis.modeltree)
            .filter(NarfAnalysis.name == aSelected)
            .first()[0]
            )
    # Pass modeltree string from NarfAnalysis to a ModelFinder constructor,
    # which will use a series of internal methods to convert the tree string 
    # to a list of model names.
    mf = ModelFinder(modeltree)
    
    session.close()
    
    return jsonify(modellist=mf.modellist)


@app.route('/update_cells')
def update_cells():
    """Update the list of cells in the cell selector after a batch
    is selected (this will cascade from an analysis selection).
    
    Also updates current batch in NarfAnalysis for current analysis.
    
    """
    
    session = Session()
    # Only get the numerals for the selected batch, not the description.
    bSelected = request.args.get('bSelected')
    aSelected = request.args.get('aSelected')

    celllist = [
            i[0] for i in 
            session.query(NarfBatches.cellid)
            .filter(NarfBatches.batch == bSelected[:3])
            .all()
            ]
    
    batchname = (
            session.query(sBatch)
            .filter(sBatch.id == bSelected[:3])
            .first()
            )
    if batchname:
        batch = str(bSelected[:3] + ': ' + batchname.name)
    else:
        batch = bSelected
    analysis = (
            session.query(NarfAnalysis)
            .filter(NarfAnalysis.name == aSelected)
            .first()
            )
    analysis.batch = batch

    session.commit()
    session.close()
    
    return jsonify(celllist=celllist)


@app.route('/update_results')
def update_results():
    """Update the results table after a batch, cell or model selection
    is changed.
    
    """
    
    session = Session()
    nullselection = """
            MUST SELECT A BATCH AND ONE OR MORE CELLS AND
            ONE OR MORE MODELS BEFORE RESULTS WILL UPDATE
            """
    
    bSelected = request.args.get('bSelected')
    cSelected = request.args.getlist('cSelected[]')
    mSelected = request.args.getlist('mSelected[]')
    colSelected = request.args.getlist('colSelected[]')
    # If no batch, cell or model is selected, display an error message.
    if (len(bSelected) == 0) or (not cSelected) or (not mSelected):
        return jsonify(resultstable=nullselection)
    # Only get numerals for selected batch.
    bSelected = bSelected[:3]
    # Use default value of 500 if no row limit is specified.
    rowlimit = request.args.get('rowLimit', 500)
    ordSelected = request.args.get('ordSelected')
    # Parse string into appropriate sqlalchemy method
    if ordSelected == 'asc':
        ordSelected = asc
    elif ordSelected == 'desc':
        ordSelected = desc
    sortSelected = request.args.get('sortSelected', 'cellid')

    # Always add cellid, batch and modelname to column lists,
    # since they are required for selection behavior.
    cols = [
            getattr(NarfResults, 'cellid'),
            getattr(NarfResults, 'modelname'),
            ]
    cols += [
            getattr(NarfResults, c) for c in colSelected
            if hasattr(NarfResults, c)
            ]
    
    # Package query results into a DataFrame
    results = psql.read_sql_query(
            Query(cols,session)
            .filter(NarfResults.batch == bSelected)
            .filter(NarfResults.cellid.in_(cSelected))
            .filter(NarfResults.modelname.in_(mSelected))
            .order_by(ordSelected(getattr(NarfResults, sortSelected)))
            .limit(rowlimit).statement,
            session.bind
            )
    resultstable = results.to_html(
            index=False, classes="table-hover table-condensed",
            )
    
    session.close()
    
    return jsonify(resultstable=resultstable)


@app.route('/update_analysis')
def update_analysis():
    """Update list of analyses after a tag and/or filter selection changes."""
    
    session = Session()
    
    tagSelected = request.args.get('tagSelected')
    statSelected = request.args.get('statSelected')
    # If special '__any' value is passed, set tag and status to match any
    # string in ilike query.
    if tagSelected == '__any':
        tString = '%%'
    else:
        tString = '%' + tagSelected + '%'
    if statSelected == '__any':
        sString = '%%'
    else:
        sString = statSelected

    analysislist = [
            i[0] for i in 
            session.query(NarfAnalysis.name)
            .filter(NarfAnalysis.tags.ilike(tString))
            .filter(NarfAnalysis.status.ilike(sString))
            .order_by(asc(NarfAnalysis.id))
            .all()
            ]
    
    session.close()
    
    return jsonify(analysislist=analysislist)


@app.route('/update_analysis_details')
def update_analysis_details():
    """Update contents of the analysis details popover when the analysis
    selection is changed.
    
    """
    
    session = Session()
    # TODO: Find a better/centralized place to store these options.
    # Columns to display in detail popup - add/subtract here if desired.
    detailcols = ['id', 'status', 'question', 'answer']
    
    aSelected = request.args.get('aSelected')
    
    cols = [
            getattr(NarfAnalysis,c) for c in detailcols 
            if hasattr(NarfAnalysis,c)
            ]
    
    # Package query results into a DataFrame
    results = psql.read_sql_query(
            Query(cols,session)
            .filter(NarfAnalysis.name == aSelected)
            .statement,
            session.bind
            )
    
    detailsHTML = """"""
    for col in detailcols:
        # Use a single line for id and status columns
        if (col == 'id') or (col == 'status'):
            detailsHTML += """
                <p><strong>%s</strong>: %s</p>
                """%(col,results.get_value(0, col))
        # Use a header + paragraph for everything else
        else:
            detailsHTML += """
                <h5><strong>%s</strong>:</h5>
                <p>%s</p>
                """%(col,results.get_value(0, col))
                    
    session.close()
    
    return jsonify(details=detailsHTML)


@app.route('/update_status_options')
def update_status_options():
    
    session = Session()
    
    statuslist = [
        i[0] for i in
        session.query(NarfAnalysis.status)
        .distinct().all()
        ]

    session.close()
    
    return jsonify(statuslist=statuslist)


@app.route('/update_tag_options')
def update_tag_options():
    
    session = Session()
    
    tags = [
        i[0].split(",") for i in
        session.query(NarfAnalysis.tags)
        .distinct().all()
        ]
    # Flatten list of lists into a single list of all tag strings
    # and remove leading and trailing whitespace.
    taglistbldupspc = [i for sublist in tags for i in sublist]
    taglistbldup = [t.strip() for t in taglistbldupspc]
    # Reform the list with only unique tags
    taglistbl = list(set(taglistbldup))
    # Finally, remove any blank tags and sort the list.
    taglist = [t for t in taglistbl if t != '']
    taglist.sort()
    
    session.close()
    
    return jsonify(taglist=taglist)
    

##############################################################################
################      edit/delete/new  functions for Analysis Editor #########
##############################################################################



#TODO: Handle Analysis Editor functions with an AJAX call instead of a form
#      submission so that the entire page doesn't have to be refreshed each
#      time - really only need to update the analysis selector's options.


@app.route('/edit_analysis', methods=['GET','POST'])
def edit_analysis():
    """Take input from Analysis Editor modal and save it to the database.
    
    Button : Edit Analysis
    
    """
    
    session = Session()
    modTime = str(datetime.datetime.now().replace(microsecond=0))
    
    eName = request.args.get('name')
    eStatus = request.args.get('status')
    eTags = request.args.get('tags')
    eQuestion = request.args.get('question')
    eAnswer = request.args.get('answer')
    eTree = request.args.get('tree')
    #TODO: add checks to require input inside form fields
    #      or allow blank so that people can erase stuff?
    
    #TODO: this requires that all analyses have to have a unique name.
    #       better way to do this or just enforce the rule?
    
    # Find out if an analysis with same name already exists.
    # If it does, grab its sql alchemy object and update it with new values,
    # so that the analysis with the same id is overwritten instead of
    # adding a new one.
    checkExists = (
            session.query(NarfAnalysis)
            .filter(NarfAnalysis.name == eName)
            .all()
            )
    if len(checkExists) > 1:
        session.close()
        return Response(
                """
                Oops! More than one analysis with the same name\
                already exists, something is wrong!
                """
                )
    elif len(checkExists) == 1:
        a = checkExists[0]
        a.name = eName
        a.status = eStatus
        a.question = eQuestion
        a.answer = eAnswer
        a.tags = eTags
        a.lastmod = modTime
        a.modeltree = eTree
    # If it doesn't exist, add new sql alchemy object with the
    # appropriate attributes, which should get assigned to a new id
    else:
        a = NarfAnalysis(
                name=eName, status=eStatus, question=eQuestion,
                answer=eAnswer, tags=eTags, batch='',
                lastmod=modTime, modeltree=eTree,
                )
        session.add(a)
    
    # For verifying correct logging - comment these out 
    # when not needed for testing.
    #print("Added the following analysis to database:")
    #print("------------------")
    #print("name:"); print(a.name)
    #print("question:"); print(a.question)
    #print("answer:"); print(a.answer)
    #print("status:"); print(a.status)
    #print("tags:"); print(a.tags)
    #print("model tree:"); print(a.modeltree)
    #print("-----------------\n\n")
    addedName = a.name
    session.commit()
    session.close()
    
    # After handling submissions, return user to main page so that it
    # refreshes with new analysis included in list    
    return jsonify(success="Analysis %s saved successfully."%addedName)


@app.route('/get_current_analysis')
def get_current_analysis():
    """Populate the Analysis Editor form with the database contents for the
    currently selected analysis.
    
    """
    
    session = Session()
    
    aSelected = request.args.get('aSelected')
    # If no analysis was selected, fill fields with blank text to
    # mimic 'New Analysis' behavior.
    if len(aSelected) == 0:
        return jsonify(
                name='', status='', tags='', question='',
                answer='', tree='',
                )
        
    a = (
        session.query(NarfAnalysis)
        .filter(NarfAnalysis.name == aSelected)
        .first()
        )
    
    session.close()
    
    return jsonify(
            name=a.name, status=a.status, tags=a.tags,
            question=a.question, answer=a.answer, tree=a.modeltree,
            )
        
    
@app.route('/check_analysis_exists')
def check_analysis_exists():
    """Check for a duplicate analysis name when an Analysis Editor form is
    submitted. If a duplicate exists, warn the user before overwriting.
    
    """
    
    session = Session()
    
    nameEntered = request.args.get('nameEntered')
    
    result = (
            session.query(NarfAnalysis)
            .filter(NarfAnalysis.name == nameEntered)
            .first()
            )
             
    if result is None:
        exists = False
    else:
        exists = True
        
    session.close()
    
    return jsonify(exists=exists)


@app.route('/delete_analysis')
def delete_analysis():
    """Delete the selected analysis from the database."""
    session = Session()
    
    success = True
    aSelected = request.args.get('aSelected')
    if len(aSelected) == 0:
        success = False
        return jsonify(success=success)
    
    result = (
            session.query(NarfAnalysis)
            .filter(NarfAnalysis.name == aSelected)
            .first()
            )
    if result is None:
        success = False
        return jsonify(success=success)

    # Leave these tests here incase accidental deletion
    # or some other issue occurs.
    # That way, the prints can be copy pasted back into a new analysis
    # form to restore the database entry.
    #print("checking for correct deletion. Deleting:")
    #print("name:"); print(result.name)
    #print("question:"); print(result.question)
    #print("answer:"); print(result.answer)
    #print("status:"); print(result.status)
    #print("tags:"); print(result.tags)
    #print("batch:"); print(result.batch)
    #print("model tree:"); print(result.modeltree)

    session.delete(result)
    session.commit()
    session.close()

    return jsonify(success=success)



####################################################################
###############     TABLE SELECTION FUNCTIONS     ##################
####################################################################



@app.route('/get_preview')
def get_preview():
    """Queries the database for the filepath to the preview image
    for the selected cell, batch and model combination(s)
    
    """
    
    session = Session()
    
    # Only get the numerals for the selected batch, not the description.
    bSelected = request.args.get('bSelected', type=str)[:3]
    cSelected = request.args.getlist('cSelected[]')
    mSelected = request.args.getlist('mSelected[]')

    path = (
            session.query(NarfResults.figurefile)
            .filter(NarfResults.batch == bSelected)
            .filter(NarfResults.cellid.in_(cSelected))
            .filter(NarfResults.modelname.in_(mSelected))
            .first()
            )
    
    if not path:
        return jsonify(filepaths=['/missing_preview'])
    
    #return jsonify(filepaths=path.figurefile)
    
    try:
        with open('/' + path.figurefile, 'r+b') as img:
            image = str(b64encode(img.read()))[2:-1]
        #return Response(image, mimetype="image/png")
        return jsonify(image=image)
    except:
        #return Response(
        #        """
        #        Image path exists in DB but
        #        image not in local storage
        #        """
        #        )
        try:
            with open(path.figurefile, 'r+b') as img:
                image = str(b64encode(img.read()))[2:-1]
            return jsonify(image=image)
        except Exception as e:
            print(e)
            with open(app.static_folder + '/lbhb_logo.png', 'r+b') as img:
                image = str(b64encode(img.read()))[2:-1]
            return jsonify(image=image)
        

@app.route('/missing_preview')
def missing_preview():
    """
    DEPRECATED
    Return an error message if no preview filepath has been stored
    in the database.
    
    """
    
    return Response('No preview image exists for this result')

@app.route('/preview/<path:filepath>')
def preview(filepath):
    """
    DEPRECATED
    Open the .png preview image at the specified path, or display
    an error message if the file is missing in local storage.
    
    """
    
    try:
        with open('/' + filepath, 'r+b') as img:
            image = img.read()
        #return Response(image, mimetype="image/png")
        return jsonify(image=image)
    except:
        #return Response(
        #        """
        #        Image path exists in DB but
        #        image not in local storage
        #        """
        #        )
        return jsonify(
                "Image path exists in DB, "
                "but image is not in local storage."
                )
        
#end