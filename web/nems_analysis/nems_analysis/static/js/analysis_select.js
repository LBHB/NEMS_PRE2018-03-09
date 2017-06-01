$(document).ready(function(){
    $("#analysisSelector").change(function(){
        // if analysis selection changes, get the value selected
        var aSelected = $("#analysisSelector").val();
                         
        // pass the value to '/update_batch' in nemsweb.py
        // get back associated batchnum and change batch selector to match
        $.ajax({
            url: $SCRIPT_ROOT + '/update_batch',
            data: { aSelected:aSelected }, 
            type: 'GET',
            success: function(data) {
                console.log("batch retrieved?: " + data.batch);
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
                console.log("modellist = " + data.modellist);
                var $models = $("#modelSelector");
                $models.empty();
                             
                $.each(data.modellist, function(modelname) {
                    console.log("modelname = " + data.modellist[modelname]);
                    $models.append($("<option></option>")
                        .attr("value", data.modellist[modelname]).text
                        (data.modellist[modelname]));
                });
            },
            error: function(error) {
                console.log(error);
            }     
        });
    });

    $("#batchSelector").change(function(){
        // TODO: update cell list when batch changes
        //       should cascade from change to analysis selection
        // if batch selection changes, get the value of the new selection
        var bSelected = $("#batchSelector").val();

        $.ajax({
            url: $SCRIPT_ROOT + '/update_cells',
            data: { bSelected:bSelected },
            type: 'GET',
            success: function(data) {
                cells = $("#cellSelector");
                cells.empty();
                console.log("new cell list = " + data.celllist)

                $.each(data.celllist, function(cell) {
                    cells.append($("<option></option>")
                        .attr("value", data.celllist[cell]).text
                        (data.celllist[cell]));                    
                });
            },
            error: function(error) {
                console.log(error);
            }    
        });
    });
     
    var colSelected = [];
    
    function updatecols(){
        var checks = document.getElementsByName('result-option[]');
        colSelected.length = 0; //empty out the options, then push the new ones
        for (var i=0; i < checks.length; i++) {
            if (checks[i].checked) {
                colSelected.push(checks[i].value);
            }
        }
    }
    
    updatecols();

    $("#batchSelector,#modelSelector,#cellSelector,.result-option,#rowLimit")
    .change(function(){
        
        updatecols();
        var bSelected = $("#batchSelector").val();
        var cSelected = $("#cellSelector").val();
        var mSelected = $("#modelSelector").val();
        var rowLimit = $("#rowLimit").val();
                         
        $.ajax({
            url: $SCRIPT_ROOT + '/update_results',
            data: { bSelected:bSelected, cSelected:cSelected, 
                   mSelected:mSelected, colSelected:colSelected,
                   rowLimit:rowLimit },
            type: 'GET',
            success: function(data) {
                //grabs whole div - replace inner html with new table?
                results = $("#tableWrapper");
                results.html(data.resultstable)
            },
            error: function(error) {
                console.log(error);
            }
        });
    });
});
        
