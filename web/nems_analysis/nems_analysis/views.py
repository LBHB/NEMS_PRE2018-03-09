from nems_analysis import app

@app.route('/')
def main_view():
    return 'Testing'
