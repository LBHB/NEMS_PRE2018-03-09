$(document).ready(function(){

    $(".plotSelect").change(updatePlot);
    function updatePlot(){
        var plotDiv = $(this).parents(".row").find(".plot-wrapper");
        var modAffected = $(this).parents(".row").attr('id');
        var plotType = $(this).val();
        
        $.ajax({
            url: $SCRIPT_ROOT + '/update_modelpane_plot',
            data: { modAffected:modAffected, plotType:plotType },
            type: 'GET',
            success: function(data){
                plotDiv.html(data.html);
            },
            error: function(error){
                console.log(error);
            }
        });
        
    }
    
    $("#updateIdx").on('click', updateIdx);
    function updateIdx(){
        var plot_stimidx = $("#changeStimIdx").val();
        var plot_dataidx = $("#changeDataIdx").val();
        
        $.ajax({
            url: $SCRIPT_ROOT + '/update_idx',
            data: { plot_stimidx:plot_stimidx, plot_dataidx:plot_dataidx },
            type: 'GET',
            success: function(data){
                $(".plotSelect").each(updatePlot);
            },
            error: function(error){
                console.log(error);       
            }
        });
    }

    $("#updateModule").on('click', updateModule);
    function updateModule(){
        var modAffected = $(this).parents(".row").attr('id');
        var fields = $(this).parents(".row").find(".editableFields").find(".input-finder");
        var values = $(this).parents(".row").find(".editableFields").find(".fieldValue");
        var fields_values = {};
        fields.each(function(i){
            fields_values[field[i].val()] = values[i].val();
        });

        $.ajax({
            url: $SCRIPT_ROOT + '/update_module',
            data: { fields_values:fields_values, modAffected:modAffected },
            type: 'GET',
            success: function(data){
                $(".plotSelect").each(updatePlot);
                console.log('success')
            },
            failure: function(error){
                console.log(error);    
            }
        })

    }


    // Not used //
    /*
    $(".selectPreset").change(updatePresets);
    function updatePresets(){
        var modAffected = $(this).parents(".row").attr('id');
        var kSelected = $(this).val();
        var parentDiv = $(this).parents(".control-group");
        
        $.ajax({
            url: $SCRIPT_ROOT + '/get_kw_defaults',
            data: { modAffected:modAffected, kSelected:kSelected },
            type: 'GET',
            success: function(data){
                kwdict = eval(data);
                for (var key in kwdict){
                    console.log("kwdict key: " + kwdict[key]);
                    console.log("just the key: " + key);
                    if (key.length == 0){
                        continue;        
                    }
                    field = parentDiv.find(
                            ".input-finder:contains(" + key + ")"
                            ).children("input");
                    field.val(kwdict[key]);
                }
            },
            error: function(error){
                console.log(error);        
            }
        });
            
    }
    */
    
});