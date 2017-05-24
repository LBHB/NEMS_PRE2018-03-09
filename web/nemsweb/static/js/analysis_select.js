$SCRIPT_ROOT = {{ request.script_root|tojson|safe }};</script>

$(document).ready(function(){
    function chooseAnalysis(){
        // if analysis selection changes, get the value selected
        var aSelected = $("#analysisSelector").val();

        // pass the value to '/update_batch' in nemsweb.py
        $.getJSON(SCRIPT_ROOT + '/update_batch', {
            aSelected: aSelected
            
        // get back batch number that maches the analysis
        }, function(newBatch) {
        // update current selection in batch selector
            $("input[name='batchnum']").val(newBatch.batchnum).change();
        });
        
        // also pass analysis value to 'update_models' in nemsweb.py
        $.getJSON(SCRIPT_ROOT + '/update_models', {
            aSelected: aSelected
            
        // get back list of models that match the analysis' modelstring
        }, function(newModels) {
        // update which models are displayed in the list
        
        // get model selector object and empty the old options
                models = $("input[name='modelnames']")
                models.empty();
                
        // for each modelname returned by nemsweb.py
        // append a new option for the select object with both value
        // and display text equal to modelname
                $.each(newModels.modellist, function(modelname) {
                    models.append($("<option></option>")
                        .attr("value", modelname).text(modelname));
                });
        });
    }
    $("input[name='batchnum']").change(){
        // TODO: update cell list when batch changes
        //       should cascade from change to analysis selection
        
        // if batch selection changes, get the value of the new selection
        var bSelected = $("input[name='batchnum']").val();
        
        // pass value to '/update_cells' in nemsweb.py
        $.getJSON(SCRIPT_ROOT + '/update_cells', {
            bSelected: bSelected
            
        // get back list of cells that match the batch selection
        }, function(newCells) {
        // update which cells are displayed in the list
                
        // get cell selector object and empty the old options
                cells = $("input[name='celllist']")
                cells.empty();
                
        // for each cellid returned, append new option
                $.each(newCells.celllist, function(cell) {
                    cells.append($("<option></option>")
                        .attr("value", cell).text(cell));                    
                });
        });
        
    }

})
        
