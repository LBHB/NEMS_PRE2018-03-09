$(document).ready(function(){
    //initializes bootstrap popover elements
    $('[data-toggle="popover"]').popover({
        trigger: 'click',
    });
    
    var analysisCheck = $("#analysisSelector").val()
    if ((analysisCheck !== "") && (analysisCheck !== undefined) && (analysisCheck !== null)){
        updateAnalysis();
    }
        
    $("#analysisSelector").change(updateAnalysis);

    function updateAnalysis(){
        // if analysis selection changes, get the value selected
        var aSelected = $("#analysisSelector").val();
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
                var $models = $("#modelSelector");
                $models.empty();
                             
                $.each(data.modellist, function(modelname) {
                    $models.append($("<option></option>")
                        .attr("value", data.modellist[modelname]).text
                        (data.modellist[modelname]));
                });
            },
            error: function(error) {
                console.log(error);
            }     
        });
    };

    $("#batchSelector").change(updateBatch);
            
    function updateBatch(){
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
                return order[i].value;
            }
        }
    }
    
    function updateSort(){
        var sort = document.getElementsByName('sort-option[]');
        for (var i=0; i < sort.length; i++) {
            if (sort[i].checked) {
                return sort[i].value;
            }
        }
    }
            
    // update at start of page, and again if changes are made
    updatecols();
    ordSelected = updateOrder();
    sortSelected = updateSort();

    $("#batchSelector,#modelSelector,#cellSelector,.result-option,#rowLimit,.order-option,.sort-option")
    .change(function(){
        
        updatecols();
        ordSelected = updateOrder();
        sortSelected = updateSort();
        
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
                results.html(data.resultstable)
            },
            error: function(error) {
                console.log(error);
            }
        });
    });

    $("#batchSelector,#modelSelector,#cellSelector,#measureSelector,#analysisSelector")
    .change(function(){
        var empty = false;
        $(".form-control").each(function() {
            if (!($(this).val()) || ($(this).val().length == 0)) {
                    empty = true;
                }
        });
        
        if (empty){
            $(".plotsub").attr('disabled','disabled');
            $("#form-warning").html("<p>Selection required for each option before submission</p>")
        } else {
            $(".plotsub").removeAttr('disabled');
            $("#form-warning").html("")
        }   
    });

    $("#analysisSelector").change(function(){
        
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
    });

    $(document).on('click','.dataframe tr',function(){
        if ($(this).hasClass('selectedRow')){
            $(this).removeClass('selectedRow');
        } else{
            $(this).addClass('selectedRow');
        }
    });

    $(document).on('click','#preview',function(e){
        var cSelected = [];
        var mSelected = [];
        var bSelected = $("#batchSelector").val();
        
        $(".dataframe tr.selectedRow").each(function(){
            cSelected.push($(this).children().eq(1).html());
        });
        $(".dataframe tr.selectedRow").each(function(){
            mSelected.push($(this).children().eq(2).html());
        });

        // only proceed if selections have been made
        if ((cSelected.length == 0) || (mSelected.length == 0)){
            alert('Must select at least one result from table')
            return false;
        }
        
        $.ajax({
            url: $SCRIPT_ROOT + '/get_preview',
            data: { cSelected:cSelected, mSelected:mSelected,
                   bSelected:bSelected },
            type: 'GET',
            success: function(data){
                // TODO: get this to open multiple windows. currently just
                // opens the first one then stops.
                for (var i=0;i<data.filepaths.length;i++){
                    window.open('preview' + data.filepaths[i],
                                'width=520','height=910');
                }
            },
            error: function(error){
                console.log(error);        
            }
        });
    });
                
    $("#clearSelected").on('click',function(){
        $(".dataframe tr.selectedRow").each(function(){
            $(this).removeClass('selectedRow');
        });
    });
});
        
