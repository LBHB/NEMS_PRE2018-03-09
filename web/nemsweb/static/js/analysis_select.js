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
                console.log("batchnum retrieved?: " + data.batchnum);
                $("select[name='batchnum']").val(data.batchnum).change();
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
                var $models = $("select[name='modelnames']");
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

    $("select[name='batchnum']").change(function(){
        // TODO: update cell list when batch changes
        //       should cascade from change to analysis selection
        // if batch selection changes, get the value of the new selection
        var bSelected = $("select[name='batchnum']").val();

        $.ajax({
            url: $SCRIPT_ROOT + '/update_cells',
            data: { bSelected:bSelected },
            type: 'GET',
            success: function(data) {
                cells = $("select[name='celllist']");
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

    $("select[name='batchnum'],select[name='modelnames'],select[name='celllist']")
    .change(function(){
        var bSelected = $("select[name='batchnum']").val();
        var cSelected = $("select[name='celllist']").val();
        var mSelected = $("select[name='modelnames']").val();
        
        $.ajax({
            url: $SCRIPT_ROOT + '/update_results',
            data: { bSelected:bSelected, cSelected:cSelected, mSelected:mSelected },
            type: 'GET',
            success: function(data) {
                //grabs whole div - replace inner html with new table?
                results = $(".table-responsive");
                results.html(data.resultstable)
            },
            error: function(error) {
                console.log(error);
            }
        });
    });
});
        
