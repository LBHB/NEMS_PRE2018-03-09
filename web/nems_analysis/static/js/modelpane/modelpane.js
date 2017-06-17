$(document).ready(function(){
        
    $(".collapse.show").each(function(){
        $(this).collapse({
            'toggle': true
        });
    });
   
    // TODO: How to get matplotlib images into the image wrappers?
    $(".plotPNG").each(function(){
        //var bytes = $(this).value
        //$(this).src = "data:image/png;base64," + bytes;
    });
    
    // TODO: AJAX calls with selection changes when parameters adjusted.
    //       Get back new plot(s) to refresh page.
    //       Just replace inner HTML of plot wrapper with new <img> ?
    
    
});