<! Template for main page/user interface >
<!doctype html>

<head>
    <title>
        Index
    </title>
    <link rel=stylesheet type=text/css href=
    "{{ url_for('static', filename='css/style.css') }}">
    
    <!-- do i need these here if i just import script from a separate file? -->
    <script type=text/javascript src=
        "{{ url_for('static', filename='js/jquery.js') }}"></script>
    <script type=text/javascript>
        $SCRIPT_ROOT = {{ request.script_root|tojson|safe }};</script>
        
</head>
<body>
    <div class=page>
        <h1>Select Analysis</h1>
        
        <!-- Collect form data for filtering selections by analysis -->
        <!-- Not yet implemented, just displays selections -->
        <form>
        <select name="analysis" size="20">
            {% for analysis in analysislist %}
                <option value="{{ analysis }}"> {{ analysis }} </option>
            {% endfor %}
        </select>
        </form>
        
    <div class=page>
        <h1>Generate Plot</ht>
        
        <!-- Collect form data for plot generation -->
        <form action="{{ url_for('handle_plot') }}", method="POST">
        <select name="plottype" size="20">
            <option value="{{ plottypelist }}"> {{ plottypelist }} </option>
        </select>
        <select name="measure" size="20">
            <option value="{{ measurelist }}"> {{ measurelist }} </option>
        </select>
        <select name="tablename" size="20">
            <option value="{{ tablelist }}"> {{ tablelist }} </option>
        </select>
        <select name="batchnum" size="20">
            {% for batch in batchlist %}
                <option value="{{ batch }}"> {{ batch }} </option>
            {% endfor %}
        </select>
        <select name="modelnameX" size="20">
            {% for model in modellist %}
                <option value="{{ model }}"> {{ model }} </option>
            {% endfor %}
        </select>
        
        <! TODO: fix so that same model name cannot be chosen from both lists>
        
        <select name="modelnameY" size="20">
            {% for model in modellist %}
                <option value="{{ model }}"> {{ model }} </option>
            {% endfor %}
        </select>
        <select name="measure" size="20">
            <option value="{{ measurelist }}"> {{ measurelist }} </option>
        </select>
        <input type="submit" value="Submit">
        </form>
    
    <div class=page>
    
        <h1>View Database Table</h1>
        
        <!-- collect form data for database query i.e. table display -->
        <form action="{{ url_for('handle_query') }}", method="POST">
        <select name="tablename" size="20">
            <option value="{{ tablelist }}"> {{ tablelist }} </option>
        </select>
        
        <select name="batchnum" size="20">
            {% for batch in batchlist %}
                <option value="{{ batch }}"> {{ batch }} </option>
            {% endfor %}
        </select>
        
        <select name="modelname" size="20">
            {% for model in modellist %}
                <option value="{{ model }}"> {{ model }} </option>
            {% endfor %}
        </select>
        <input type="submit" value="Submit">
        </form>
        
    </div>
    <div class=footer>
    </div>
</body>
</html>