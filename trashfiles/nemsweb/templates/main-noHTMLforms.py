<! Template for main page/user interface >
<!doctype html>

<head>
    <title>
        Index
    </title>
    <link rel=stylesheet type=text/css href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <div class=page>
        <h1>Index</h1>
        
        <select id="chooseTable" size="20" onchange="updateValue('tablename')">
            <option value="{{ tablelist }}"> {{ tablelist }} </option>
        </select>
        
        <select id="chooseBatch" size="20" onchange="updateValue('batchnum')">
            {% for batch in batchlist %}
                <option value="{{ batch }}"> {{ batch }} </option>
            {% endfor %}
        </select>
        
        <select id="chooseModel" size="20" onchange="upateValue('modelname')">
            {% for model in modellist %}
                <option value="{{ model }}"> {{ model }} </option>
            {% endfor %}
        </select>
        
        <a href="" id="queryurl"> SUBMIT </a>
        
        <p id="test"></p>
        
        <script>
        var tablename = 'NarfResults'
        var batchnum = '0'
        var modelname = ''
        
        function updateValue(varName) {
            if (varName === 'tablename'){
                tablename = document.getElementById("chooseTable").value;
            }
            if (varName === 'batchnum'){
                batchnum = document.getElementById("chooseBatch").value;
            }
            if (varName === 'modelname'){
                modelname = document.getElementById("chooseModel").value;    
            }
            
            document.getElementById("test").innerHTML = "tablename: " + tablename +
            ", batchnum: " + batchnum + ", modelname: " + modelname;
        }
            
        document.getElementById("queryurl").href = "http://127.0.0.1:5000/query/tablename="
        + tablename + "/batchnum=" + batchnum + "/modelname=" + modelname;
        </script>
        
    </div>
    <div class=footer>
    </div>
</body>
</html>