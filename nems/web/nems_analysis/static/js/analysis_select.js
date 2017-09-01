$(document).ready(function(){
        
    // TODO: Split this up into multile .js files? getting a bit crowded in here,
    // could group by functionality at this point.    

    namespace = '/py_console'
    var socket = io.connect(
            location.protocol + '//'
            + document.domain + ':' 
            + location.port + namespace,
            {'timeout':0}
            );
            
    socket.on('connect', function() {
       console.log('socket connected');
    });
    
    socket.on('console_update', function(msg){
        var color = '';
        // TODO: add special color markers here for other messages?
        if ((msg.data === 'Login required') || (msg.data === 'Fit failed.')){
            color = 'style="color: red"';
        }
        $('#py_console').prepend(
                "<p class='py_con_msg'" + color + ">" + msg.data + "</p>"
                );
    });
    
    // use this in place of console.log to send to py_console
    function py_console_log(message){
        var color = '';
        // TODO: add special color markers here for other messages?
        if ((message === 'Login required') || (message === 'Fit failed.')){
            color = 'style="color: red"';
        }
        $('#py_console').prepend(
                "<p class='py_con_msg'" + color + ">" + message + "</p>"
                );
    }

    //initializes bootstrap popover elements
    $('[data-toggle="popover"]').popover({
        trigger: 'click',
    });
     
    function sizeDragDisplay(){
        display = $("#displayOuter");
        display.resizable({
            handles: "n, w, e, s"
        });
        display.draggable();
    }
        
    function sizeDragTable(){
        table = $("#resultsArea");
        table.resizable({
            handles: "n, w, e, s"     
        });
        table.draggable();
    }

    $("#selectArea").resizable({
        handles: "n, w, e, s"        
    });
    $("#py_console").resizable({
        handles: "n, s"        
    });
    //$("#py_console").draggable();
    $("#selectArea").draggable();

    sizeDragDisplay();
    sizeDragTable();
    //drags start out disabld until alt is pressed
    $(".dragToggle").draggable('disable');
    
    $("#enableDraggable").on('click', function(){
        $(".dragToggle").draggable('enable');
        return false;
    });
    $("#disableDraggable").on('click', function(){
        $(".dragToggle").draggable('disable');
        return false;
    });
                 
    function initTable(table){
        // turned off DataTable for now since it wasn't being used for much.
        return false;
        // Called any time results is updated -- set table options here
        table.DataTable({
            paging: false,
            search: {
                "caseInsensitive": true,
            },
            //responsive: true,
            // TODO: add select option to replace custom one
            // (can add ctrl/shift click etc)
        });
    }

    /*
    var saved_selections = new Object()
    function get_saved_selections(){
        $.ajax({
            url: $SCRIPT_ROOT + '/get_saved_selections'
            data: {},
            type: 'GET',
            success: function(data){
                saved_selections = data.selections;
            }
            error: function(error){
                console.log(error);
            }
        });
    }

    function update_selections(){
        get_saved_selections();

        if (saved_selections.hasOwnProperty('tag')){
            // iterate through tag options and select the one that matches,
            // set others to unchecked
        }
        if (saved_selections.hasOwnProperty('status')){
            // iterate through status options and select the one that matches,
            // set others to unchecked
        }
        if (saved_selections.hasOwnProperty('analysis')){
            $("#analysisSelector").val(saved_selections['analysis']);
        }
        if (saved_selections.hasOwnProperty('plot_measure')){
            $("#measureSelector").val(saved_selections['plot_measure']);
        }
        if (saved_selections.hasOwnProperty('plot_type')){
            $("#plotTypeSelector").val(saved_selections['plot_type']);
        }
        if (saved_selections.hasOwnProperty('onlyFair')){
            if ((int)saved_selections['onlyFair'] === 1){
                document.getElementById('onlyFair').checked = true;
            } else{
                document.getElementById('onlyFair').checked = false;
            }
        }
        if (saved_selections.hasOwnProperty('includeOutliers')){
            if ((int)saved_selections['includeOutliers'] === 1){
                document.getElementById('includeOutliers').checked = true;
            } else{
                document.getElementById('includeOutliers').checked = false;
            }
        }
        if (saved_selections.hasOwnProperty('snr')){
            snr = saved_selections['snr'];
        }
        if (saved_selections.hasOwnProperty('snri')){
            snri = saved_selections['snri'];
        }
        if (saved_selections.hasOwnProperty('iso')){
            iso = saved_selections['iso'];
        }
        if (saved_selectoins.hasOwnProperty('table_cols')){
            // iterate through dropdown div -- check matching options
        }
        if (saved_selections.hasOwnProperty('sort_options')){
            // check either ascending or descending
            // iterate through other options, check matches.
        }
        if (saved_selections.hasOwnProperty('row_limit')){
            $("#rowLimit").val(saved_selections['row_limit']);
        }
    }
    */
    
    var analysisCheck = document.getElementById("analysisSelector").value;
    if ((analysisCheck !== "") && (analysisCheck !== undefined) && (analysisCheck !== null)){
        updateBatchModel();
        updateAnalysisDetails();
    }
    
    
    $("#selectAllCells").on('click', selectCellsCheck);
    function selectCellsCheck(){
        var cellOptions = document.getElementsByName("cellOption[]");
        for (var i=0;i<cellOptions.length;i++){
            if (document.getElementById("selectAllCells").checked){
                cellOptions[i].selected = true;
            } else{
                cellOptions[i].selected = false;
            }
        }
        updateResults();                           
    }

    $("#selectAllModels").on('click', selectModelsCheck);
    function selectModelsCheck(){
        var modelOptions = document.getElementsByName("modelOption[]");
        for (var i=0;i<modelOptions.length;i++){
            if (document.getElementById("selectAllModels").checked){
                modelOptions[i].selected = true;
            } else{
                modelOptions[i].selected = false;
            }
        }
        updateResults();
    }
    
    
    $("#analysisSelector").change(updateBatchModel);

    function updateBatchModel(){
        // if analysis selection changes, get the value selected
        var aSelected = $("#analysisSelector").val();
        saved_selections.analysis = aselected
        // pass the value to '/update_batch' in nemsweb.py
        // get back associated batchnum and change batch selector to match
        $.ajax({
            url: $SCRIPT_ROOT + '/update_batch',
            data: { aSelected:aSelected }, 
            type: 'GET',
            success: function(data) {
                $("#batchSelector").val(data.batch).change();
            },
            error: function(error) {
                console.log(error);
            }
        });
        // also pass analysis value to 'update_models' in nemsweb.py
        $.ajax({
            url: $SCRIPT_ROOT + '/update_models',
            data: { aSelected:aSelected }, 
            type: 'GET',
            success: function(data) {
                var models = $("#modelSelector");
                models.empty();
                             
                $.each(data.modellist, function(modelname) {
                    models.append($("<option></option>")
                        .attr("value", data.modellist[modelname])
                        .attr("name","modelOption[]")
                        .text(data.modellist[modelname]));
                });
                    
                selectModelsCheck();
            },
            error: function(error) {
                console.log(error);
            }     
        });
    };

    $("#batchSelector").change(updateCells);

    function updateCells(){
        // TODO: update cell list when batch changes
        //       should cascade from change to analysis selection
        // if batch selection changes, get the value of the new selection
        var bSelected = $("#batchSelector").val();
        var aSelected = $("#analysisSelector").val();

        $.ajax({
            url: $SCRIPT_ROOT + '/update_cells',
            data: { bSelected:bSelected, aSelected:aSelected },
            type: 'GET',
            success: function(data) {
                cells = $("#cellSelector");
                cells.empty();
                
                $.each(data.celllist, function(cell) {
                    cells.append($("<option></option>")
                        .attr("value", data.celllist[cell])
                        .attr("name","cellOption[]")
                        .text(data.celllist[cell]));                    
                });
                    
                selectCellsCheck();
            },
            error: function(error) {
                console.log(error);
            }    
        });
    };
     
    // initialize display option variables
    var colSelected = [];
    var ordSelected;
    var sortSelected;
    
    // update function for each variable
    function updatecols(){
        var checks = document.getElementsByName('result-option[]');
        colSelected.length = 0; //empty out the options, then push the new ones
        for (var i=0; i < checks.length; i++) {
            if (checks[i].checked) {
                colSelected.push(checks[i].value);
            }
        }
    }
    
    function updateOrder(){
        var order = document.getElementsByName('order-option[]');
        for (var i=0; i < order.length; i++) {
            if (order[i].checked) {
                ordSelected = order[i].value;
                return false;
            }
        }
    }
    
    function updateSort(){
        var sort = document.getElementsByName('sort-option[]');
        for (var i=0; i < sort.length; i++) {
            if (sort[i].checked) {
                sortSelected = sort[i].value;
                return false;
            }
        }
    }
            
    // update at start of page, and again if changes are made
    updatecols();
    ordSelected = updateOrder();
    sortSelected = updateSort();

    function addLinksToTable(){
        // Iterate through each table row and convert each result
        // to a link to detailed info for that result
        var $table = $("#tableWrapper").children('table');
        $table.find('tbody').find('tr').each(function(){
            var cellid = $(this).children().eq(0).html();
            var modelname = $(this).children().eq(1).html();
            var cell_link = $SCRIPT_ROOT + '/cell_details/';
            var model_link = $SCRIPT_ROOT + '/model_details/';
            $(this).children().eq(0).html(
                    "<a href='" + cell_link + cellid + "' target='_blank'"
                    + "id='" + cellid + "'>" + cellid + "</a>"
                    );
            $(this).children().eq(1).html(
                    "<a href='" + model_link + modelname + "' target='_blank'"
                    + "id='" + modelname + "'>" + modelname + "</a>"
                    );
        });
    }
    
    $("#modelSelector,#cellSelector,.result-option,#rowLimit,.order-option,.sort-option")
    .change(updateResults);
    function updateResults(){
        
        updatecols();
        updateOrder();
        updateSort();
        
        var bSelected = $("#batchSelector").val();
        var cSelected = $("#cellSelector").val();
        var mSelected = $("#modelSelector").val();
        var rowLimit = $("#rowLimit").val();
                         
        $.ajax({
            url: $SCRIPT_ROOT + '/update_results',
            data: { bSelected:bSelected, cSelected:cSelected, 
                   mSelected:mSelected, colSelected:colSelected,
                   rowLimit:rowLimit, ordSelected:ordSelected,
                   sortSelected:sortSelected },
            type: 'GET',
            success: function(data) {
                //grabs whole div - replace inner html with new table?
                results = $("#tableWrapper");
                //results.resizable("destroy");
                //results.draggable("destroy");
                results.html(data.resultstable)
                //sizeDragTable();
                var table = results.children("table");
                initTable(table);
                addLinksToTable();
            },
            error: function(error) {
                console.log(error);
            }
        });
    }
    

    ////////////////////////////////////////////////////////////////////////
    //         Analysis details, tags/status, edit/delete/new             //
    ////////////////////////////////////////////////////////////////////////


    $("#analysisSelector").change(updateAnalysisDetails);
            
    function updateAnalysisDetails(){
        var aSelected = $("#analysisSelector").val();
        
        $.ajax({
            url: $SCRIPT_ROOT + '/update_analysis_details',
            data: { aSelected:aSelected },
            type: 'GET',
            success: function(data) {
                $("#analysisDetails").attr("data-content",data.details)
                $("#analysisDetails").attr("title",aSelected)
            },
            error: function(error) {
                console.log(error);
            }
        });
    }
    
    var tagSelected;
    var statSelected;
    
    function updateTag(){
        var tags = document.getElementsByName('tagOption[]');
        for (var i=0; i < tags.length; i++) {
            if (tags[i].checked) {
                tagSelected = tags[i].value;
                saved_selections.tag = tags[i].value;
                return false;
            }
        }
    }
    
    function updateStatus(){
        var status = document.getElementsByName('statusOption[]');
        for (var i=0; i < status.length; i++) {
            if (status[i].checked) {
                statSelected = status[i].value;
                saved_selections.status = tags[i].status;
                return false;
            }
        }
    }
    
    updateTag();
    updateStatus();
    updateAnalysis();
    $(".tagOption, .statusOption").change(updateAnalysis);
    
    function updateAnalysis(){
        updateTag();
        updateStatus();

        $.ajax({
           url: $SCRIPT_ROOT + '/update_analysis',
           data: { tagSelected:tagSelected, statSelected:statSelected },
           type: 'GET',
           success: function(data){
                analyses = $("#analysisSelector");
                analyses.empty();
                
                $.each(data.analysislist, function(analysis) {
                    analyses.append($("<option></option>")
                        .attr("value", data.analysislist[analysis])
                        .text(data.analysislist[analysis]));
                });
                    
                $("#analysisSelector").val($("#analysisSelector option:first").val()).change();
           },
           error: function(error){
                console.log(error);
           }
        });
    }
            
    function updateTagOptions(){
        $.ajax({
           url: $SCRIPT_ROOT + '/update_tag_options',
           type: 'GET',
           success: function(data){
               tags = $("#tagFilters")
               tags.empty();
               tags.append(
                        "<ul class='list-unstyled col-sm-6'>"
                        + "<li>"
                        + "<input class='statusOption' name='statusOption[]' type="
                        + "'radio' value='__any' checked>"
                        + "<span class='lbl'> Any </span>"
                        + "</li>"
                        )
               
               $.each(data.taglist, function(tag){
                   if (tag%12 == 0){
                       tags.append(
                                "</ul><ul class='list-unstyled col-sm-6'>"
                                )        
                   }
                   
                   tags.append(
                           "<li>"
                            + "<input class='tagOption'"
                            + "name='tagOption[]' type='radio' value="
                            + data.taglist[tag] + "'" + ">"
                            + "<span class='lbl'>" + data.taglist[tag] + "</span>"
                            + "</li>"
                            )
               });
               tags.append("</ul>");    
            
           },
           error: function(error){
               console.log(error);
           }        
        });
    }
    
    function updateStatusOptions(){
        $.ajax({
           url: $SCRIPT_ROOT + '/update_status_options',
           type: 'GET',
           success: function(data){
               statuses = $("#statusFilters")
               statuses.empty();
               statuses.append(
                        "<li>"
                        + "<input class='statusOption' name='statusOption[]' type="
                        + "'radio' value='__any' checked>"
                        + "<span class='lbl'> Any </span>"
                        + "</li>"
                        )
               
               $.each(data.statuslist, function(status){
                   if (status == 0){
                       var checked = 'checked'
                   } else{
                       var checked = '' 
                   }
                   statuses.append(
                            "<li>"
                            + "<input class='tagOption'"
                            + "name='tagOption[]' type='radio' value="
                            + data.statuslist[status] + "'" + ">"
                            + "<span class='lbl'>" + data.statuslist[status]
                            + "</span>"
                            + "</li>"
                            )
               });
           },
           error: function(error){
               console.log(error);
           }        
        });
    }
    
    $("#newAnalysis").on('click',newAnalysis);
    
    function newAnalysis(){
        $("[name='editName']").val('');
        $("[name='editStatus']").val('');
        $("[name='editTags']").val('');
        $("[name='editQuestion']").html('');
        $("[name='editAnswer']").html('');
        $("[name='editTree']").val('');
    }
    
    $("#editAnalysis").on('click',editAnalysis);
    
    function editAnalysis(){
        // get current analysis selection
        // ajax call to get back name, tags, question, etc
        // fill in content of editor modal with response
        
        var aSelected = $("#analysisSelector").val();
        
        $.ajax({
            url: $SCRIPT_ROOT + '/get_current_analysis',
            data: { aSelected:aSelected },
            type: 'GET',
            success: function(data){
                $("[name='editName']").val(data.name);
                $("[name='editStatus']").val(data.status);
                $("[name='editTags']").val(data.tags);
                $("[name='editQuestion']").html(data.question);
                $("[name='editAnswer']").html(data.answer);
                $("[name='editTree']").val(data.tree);
            },
            error: function(error){
                console.log(error);        
            }                
        });
    }
    
    $("#submitEditForm").on('click',verifySubmit);
    
    
    // TODO: change this to run nested ajax call instead of .submit()
    // so that entire page doesn't have to refresh afterward. should only have
    // to call updateAnalysis() to refresh the list for analysis selector.
    
    function verifySubmit(){
        var nameEntered = $("[name='editName']").val();
        
        $.ajax({
           url: $SCRIPT_ROOT + '/check_analysis_exists',
           data: { nameEntered:nameEntered },
           type: 'GET',
           success: function(data){
                if (data.exists){
                    alert("WARNING: An analysis by the same name already exists.\n" +
                          "Submitting this form without changing the name will\n" +
                          "overwrite the existing analysis entry!");
                }
                
                if(confirm("ATTENTION: This will save the entered information to the\n" +
                            "database, potentially overwriting previous settings.\n" +
                            "Are you sure you want to continue?")){
                    submitAnalysis();
                                  
                } else{
                    return false;
                }
                $("#analysisSelector").val(nameEntered);

            },
           error: function(error){
                   
            }
        });
    }
                
    function submitAnalysis(){
        var name = $("[name='editName']").val();
        var status = $("[name='editStatus']").val();
        var tags = $("[name='editTags']").val();
        var question = $("[name='editQuestion']").val();
        var answer = $("[name='editAnswer']").val();
        var tree = $("[name='editTree']").val();
        
        $.ajax({
           url: $SCRIPT_ROOT + '/edit_analysis',
           data: { name:name, status:status, tags:tags,
                  question:question, answer:answer, tree:tree },
           type: 'GET',
           success: function(data){
               $("#analysisEditorModal").modal('hide')
               py_console_log(data.success);
               updateAnalysis();
           },
           error: function(error){
               console.log(error)
           }
        });
    }

    $("#deleteAnalysis").on('click',deleteAnalysis);
    
    function deleteAnalysis(){
            
        var aSelected = $("#analysisSelector").val();
        reply = confirm("WARNING: This will delete the database entry for the selected " +
                "analysis. \n\n!!   THIS CANNOT BE UNDONE   !!\n\n" +
                "Are you sure you wish to delete this analysis:\n" +
                aSelected);
        
        if (reply){
            $.ajax({
                url: $SCRIPT_ROOT + '/delete_analysis',
                data: { aSelected:aSelected },
                
                // TODO: should use POST here? but was causing issues
                
                type: 'GET',
                success: function(data){
                    if (data.success){
                        py_console_log(aSelected + " successfully deleted.");
                        updateAnalysis();
                        //updateTagOptions();
                        //updateStatusOptions();
                    } else{
                        py_console_log("Something went wrong - unable to delete:\n" + aSelected);
                        return false;
                    }
                },
                error: function(error){
                    console.log(error);
                }
            });
        } else{
            return false;        
        }
    }
    
    
    
    ///////////////////////////////////////////////////////////////////////
    //     table selection, preview, strf, inspect                       //
    ///////////////////////////////////////////////////////////////////////
    
    
    CTRL = false;
    // TODO: implement shift select
    SHIFT = false;
    UPPER = -1;
    
    $(document).keydown(function(event){
        if (event.ctrlKey){
            CTRL = true;        
        }
        if (event.shiftKey){
            SHIFT = true;
        }
    });
    $(document).keyup(function(event){
        CTRL = false;
        SHIFT = false;
    });
    
    $(document).on('click','.dataframe tbody tr',function(){
        if ($(this).hasClass('selectedRow')){
            $(this).removeClass('selectedRow');
        } else{
            //comment this out for multi-select
            if (!CTRL){
                $(this).parents('tbody').find('.selectedRow').each(function(){
                    $(this).removeClass('selectedRow');
                });
            }
            $(this).addClass('selectedRow');
            if (document.getElementById("autoPreview").checked){
                refreshPreview();
            }
        }
    });

    $(document).on('click','#preview', refreshPreview);
    function refreshPreview(){
        var cSelected = [];
        var mSelected = [];
        var bSelected = $("#batchSelector").val();
        
        $(".dataframe tr.selectedRow").each(function(){
            cSelected.push($(this).children().eq(0).children('a').attr('id'));
        });
        $(".dataframe tr.selectedRow").each(function(){
            mSelected.push($(this).children().eq(1).children('a').attr('id'));
        });

        // only proceed if selections have been made
        if ((cSelected.length == 0) || (mSelected.length == 0)){
            py_console_log('Must select at least one result from table')
            return false;
        }
        
        $.ajax({
            url: $SCRIPT_ROOT + '/get_preview',
            data: { cSelected:cSelected, mSelected:mSelected,
                   bSelected:bSelected },
            type: 'GET',
            success: function(data){
                //$("#displayWrapper").resizable("destroy");
                //$("displayWrapper").draggable("destroy");
                $("#displayWrapper").html(
                        '<img id="preview_image" src="data:image/png;base64,'
                        + data.image + '" />'
                        );
                //sizeDragDisplay();
            },
            error: function(error){
                console.log(error);        
            }
        });
    };
                
    $("#clearSelected").on('click',function(){
        $(".dataframe tr.selectedRow").each(function(){
            $(this).removeClass('selectedRow');
        });
    });
                
    $("#strf").on('click',function(){
        py_console_log("STRF Function not yet implemented");
        //return strf plots ala narf_analysis
        //low priority
    });
                
                
                
                
    //////////////////////////////////////////////////////////////////////            
    //               Model fitting, inspect                             //
    //////////////////////////////////////////////////////////////////////
    
    
    
    function addLoad(){
        $('#loadingPopup').css('display', 'block');
    }
    function removeLoad(){
        $('#loadingPopup').css('display', 'none');
    }
    
    $("#toggleFitOp").on('click',function(){
        var fod = $('#fitOpDiv');
        
        if (fod.css('display') === 'block'){
            fod.css('display', 'none');
            
        } else if (fod.css('display') === 'none'){
            fod.css('display', 'block');
            
        } else{
            return false;        
        }
    })
    
    $("#fitSingle").on('click',function(){   
        var bSelected = $("#batchSelector").val();
        var cSelected = $("#cellSelector").val();
        var mSelected = $("#modelSelector").val();
        
        if ((bSelected === null) || (bSelected === undefined) || 
                (bSelected.length == 0)){
            py_console_log('Must select a batch')
            return false;
        }
        if ((cSelected.length > 1) || (mSelected.length > 1) || (cSelected.length
            == 0) || (mSelected.length == 0)){
            py_console_log('Must select one model and one cell')
            return false;
        }
        
        if (!(confirm(
                "Preparing to fit selection -- this may take several minutes."
              + "Web interface will be disabled until fit is complete."
              + "\n\nSelect OK to continue."))){
            return false;
        }
        // TODO: insert confirmation box here, with warning about waiting for
        //          fit job to finish
        
        addLoad();
                
        py_console_log("Sending fit request to server. Success or failure will be"
                    + "reported here when job is finished.")
        
        $.ajax({
            url: $SCRIPT_ROOT + '/fit_single_model',
            data: { bSelected:bSelected, cSelected:cSelected,
                       mSelected:mSelected },
            // TODO: should use POST maybe in this case?
            type: 'GET',
            timeout: 0,
            success: function(data){
                py_console_log("Fit finished.\n"
                      + "r_test: " + data.r_est + "\n"
                      + "r_val: " + data.r_val + "\n"
                      + "Click 'inspect' to browse model");
                
                updateResults();
                removeLoad();
                //open preview in new window like the preview button?
                //then would only have to pass file path
                //window.open('preview/' + data.preview,'width=520','height=910')
            },
            error: function(error){
                console.log("Fit failed");
                console.log(error);
                removeLoad();
            },
        });
    });
        
                
    $("#enqueue").on('click',function(){  
        var bSelected = $("#batchSelector").val();
        var cSelected = $("#cellSelector").val();
        var mSelected = $("#modelSelector").val();
        var forceRerun = 0;
        
        if (document.getElementById('forceRerun').checked){
            forceRerun = 1;
        }
        
        if ((bSelected === null) || (bSelected === undefined) || 
                (bSelected.length == 0)){
            py_console_log('Must select a batch')
            return false;
        }
        if ((cSelected.length == 0) || (mSelected.length == 0)){
            py_console_log('Must select at least one model and at least one cell')
            return false;
        }
        
        var total = cSelected.length * mSelected.length;
        if (!(confirm('This will add ' + total + ' models to the queue.'
                      + '\n\nAre you sure you wish to continue?'))){
            return false;
        }
        
        addLoad();
        py_console_log("Sending fit request for each combination - please wait...");
                      
        $.ajax({
            url: $SCRIPT_ROOT + '/enqueue_models',
            data: { bSelected:bSelected, cSelected:cSelected,
                   mSelected:mSelected, forceRerun },
            // TODO: should POST be used in this case?
            type: 'GET',
            success: function(result){
                py_console_log(result);
                removeLoad();
            },
            error: function(error){
                console.log(error)   
                removeLoad();
            }
        });
        //communicates with daemon to queue model fitting for each selection on cluster,
        //using similar process as above but for multiple models and no
        //dialogue displayed afterward
        
        //open separate window/tab for additional specifications like priority?
    });
        
    $("#inspect").on('click',function(){
        //pull from results table
        var cSelected = [];
        var mSelected = [];
        var bSelected = $("#batchSelector").val();
        
        $(".dataframe tr.selectedRow").each(function(){
            cSelected.push($(this).children().eq(0).children('a').attr('id'));
            if (cSelected.length > 0){
                return false;
            }
        });
        $(".dataframe tr.selectedRow").each(function(){
            mSelected.push($(this).children().eq(1).children('a').attr('id'));
            if (mSelected.length > 0){
                return false;
            }
        });
 
        if ((cSelected.length === 0) || (mSelected.length === 0)){
            py_console_log("Must choose one cell and one model from the table.");
            return false;
        }
        
        var form = document.getElementById("modelpane");
        var batch = document.getElementById("p_batch");
        var cellid = document.getElementById("p_cellid");
        var modelname = document.getElementById("p_modelname");
        
        batch.value = bSelected;
        cellid.value = cSelected;
        modelname.value = mSelected;
        
        form.submit();
    });
        
    
    ////////////////////////////////////////////////////////////////////
    //////////////////////  PLOTS //////////////////////////////////////
    ////////////////////////////////////////////////////////////////////
    
    
    $("#togglePlotOp").on('click',function(){
        var pow = $('#plotOpWrapper');
        
        if (pow.css('display') === 'block'){
            pow.css('display', 'none');
            
        } else if (pow.css('display') === 'none'){
            pow.css('display', 'block');
            
        } else{
            return false;        
        }
    })
    
    /*
    // Default values -- based on 'good' from NarfAnalysis > filter_cells
    if saved_selections.hasOwnProperty('snr'){
        var snr = saved_selections['snr'];
    } else{
        var snr = $("#default_snr").val();
    }
    if saved_selections.hasOwnProperty('iso'){
        var iso = saved_selections['iso'];
    } else{
        var iso = $("#default_iso").val();
    }
    if saved_selections.hasOwnProperty('snri'){
        var snri = saved_selections['snri'];
    } else{
        var snri = $("#default_snri").val();
    }     
    */

    var snr = $("#default_snr").val();
    var iso = $("#default_iso").val();
    var snri = $("#default_snri").val();

    $("#plotOpSelect").val('snri');
    $("#plotOpVal").val(snri); 
    
    $("#plotOpSelect").change(updatePlotOpVal);
    function updatePlotOpVal(){
        var select = $("#plotOpSelect");
        var opVal = $("#plotOpVal");
        var getVal = 0.0;
        
        if (select.val() === 'snr'){
            getVal = snr;        
        }
        if (select.val() === 'iso'){
            getVal = iso;                
        }
        if (select.val() === 'snri'){
            getVal = snri;        
        }
        opVal.val(getVal);
    }

    $("#plotOpVal").change(updateOpVariable);
    function updateOpVariable(){
        var select = $("#plotOpSelect");
        var opVal = $("#plotOpVal");
        var setVal = opVal.val();
        
        if (select.val() === 'snr'){
            snr = setVal;       
        }
        if (select.val() === 'iso'){
            iso = setVal;                
        }
        if (select.val() === 'snri'){
            snri = setVal;       
        }
    }
     
        
    $("#submitPlot").on('click', getNewPlot);
    function getNewPlot(){
        var plotDiv = $("#displayWrapper");
        
        var plotType = $("#plotTypeSelector").val();
        var bSelected = $("#batchSelector").val();
        var cSelected = $("#cellSelector").val();
        var mSelected = $("#modelSelector").val();
        var measure = $("#measureSelector").val();
        var onlyFair = 0;
        if (document.getElementById("onlyFair").checked){
            onlyFair = 1;
        }
        var includeOutliers = 0;
        if (document.getElementById("includeOutliers").checked){
            includeOutliers = 1;
        }
        var plotNewWindow = 0;
        if (document.getElementById("plotNewWindow").checked){
            plotNewWindow = 1;        
        }
        
        addLoad();
        $.ajax({
            url: $SCRIPT_ROOT + '/generate_plot_html',
            data: { plotType:plotType, bSelected:bSelected, cSelected:cSelected,
                    mSelected:mSelected, measure:measure, onlyFair:onlyFair,
                    includeOutliers:includeOutliers, iso:iso, snr:snr, 
                    snri:snri, plotNewWindow:plotNewWindow },
            type: 'GET',
            success: function(data){
                if (data.hasOwnProperty('script')){
                    if(plotNewWindow){
                        var w = window.open(
                                $SCRIPT_ROOT + '/plot_window',
                                //"_blank",
                                //"width=600, height=600"
                                )
                        $(w.document).ready(function(){
                            w.$(w.document.body).append(data.script + data.div);
                        });
                    } else{
                        $("#statusReportWrapper").html('');
                        plotDiv.html(data.script + data.div);  
                    }
                }
                if (data.hasOwnProperty('html')){
                    if(plotNewWindow){
                        var w = window.open(
                                $SCRIPT_ROOT + '/plot_window',
                                //"_blank",
                                //"width=600, height=600" 
                                )
                        $(w.document).ready(function(){
                            w.$(w.document.body).append(data.html);
                        });
                    } else{
                        $("#statusReportWrapper").html('');
                        plotDiv.html(data.html);
                    }
                }
                removeLoad();
            },
            error: function(error){
                console.log(error)
                removeLoad();
            }
        });
    }
           
    $("#submitCustom").on('click', getCustomScript);
    function getCustomScript(){
        var scriptName = $("#customSelector").val();
        var bSelected = $("#batchSelector").val();
        var cSelected = $("#cellSelector").val();
        var mSelected = $("#modelSelector").val();
        var measure = $("#measureSelector").val();
        var onlyFair = 0;
        if (document.getElementById("onlyFair").checked){
            onlyFair = 1;
        }
        var includeOutliers = 0;
        if (document.getElementById("includeOutliers").checked){
            includeOutliers = 1;
        }
        var plotNewWindow = 0;
        if (document.getElementById("plotNewWindow").checked){
            plotNewWindow = 1;        
        }
        
        addLoad();
        $.ajax({
            url: $SCRIPT_ROOT + '/run_custom',
            data: { scriptName:scriptName, bSelected:bSelected,
                    cSelected:cSelected, mSelected:mSelected, measure:measure,
                    onlyFair:onlyFair, includeOutliers:includeOutliers,
                    iso:iso, snr:snr, snri:snri },
            type: 'GET',
            success: function(data){
                if(plotNewWindow){
                    var w = window.open(
                        $SCRIPT_ROOT + '/plot_window',
                        //"_blank",
                        //"width=600, height=600" 
                        )
                    $(w.document).ready(function(){
                        w.$(w.document.body).append(data.html);
                    });
                } else{
                    $("#statusReportWrapper").html('');
                    $("#displayWrapper").html(data.html);
                    }
                removeLoad();
            },
            error: function(error){
                console.log(error)
                removeLoad();
            }
        });
    }

    $("#reloadScripts").on('click', reloadScripts);
    function reloadScripts(){
        // call scan_for_scripts again in python then,
        // then repopulate the select element with new list
        $.ajax({
            url: $SCRIPT_ROOT + '/reload_scripts',
            data: {},
            type: 'GET',
            success: function(data){
                scripts = $("#customSelector");
                scripts.empty();

                $.each(data.scriptlist, function(sName){
                    scripts.append($("<option></option>")
                        .attr("value", data.scriptlist[sName])
                        .attr("name", "scriptOption[]")
                        .text(data.scriptlist[sName])
                        );
                });
            },
            error: function(error){
                console.log(error)
            }
        })
            var models = $("#modelSelector");
                models.empty();
                             
                $.each(data.modellist, function(modelname) {
                    models.append($("<option></option>")
                        .attr("value", data.modellist[modelname])
                        .attr("name","modelOption[]")
                        .text(data.modellist[modelname]));
                });
    }

             
    $("#batchPerformance").on('click', batchPerformance);
    function batchPerformance(){
        var bSelected = $("#batchSelector").val();
        var mSelected = $("#modelSelector").val();
        var cSelected = $("#cellSelector").val();
        var findAll = 0;
        if (document.getElementById("findAll").checked){
            findAll = 1;
        }
        
        var formInfo = document.getElementById('batchPerfForm');
        formInfo.bSelected.value = bSelected;
        formInfo.cSelected.value = cSelected;
        formInfo.mSelected.value = mSelected;
        formInfo.findAll.value = findAll;
        
        formInfo.submit();
    }
    
    $("#fitReport").on('click', fitReport);
    function fitReport(){
        var bSelected = $("#batchSelector").val();
        var mSelected = $("#modelSelector").val();
        var cSelected = $("#cellSelector").val();
        
        addLoad();
        $.ajax({
            url: $SCRIPT_ROOT + '/fit_report',
            data: { bSelected:bSelected, mSelected:mSelected,
                    cSelected:cSelected },
            type: 'GET',
            success: function(data){
                $("#statusReportWrapper").html('');
                $("#displayWrapper").html(
                        '<img id="preview_image" src="data:image/png;base64,'
                        + data.image + '" />'
                        );
                removeLoad();
            },
            error: function(error){
                console.log(error);
                removeLoad();
            }
        });

        // submit as form for new tab
        /*
        var formInfo = document.getElementById('fitRepForm');
        formInfo.bSelected.value = bSelected;
        formInfo.cSelected.value = cSelected;
        formInfo.mSelected.value = mSelected;
        
        formInfo.submit();
        */  
    }
});
        
