""" Forms for uploading data to the server. """

from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, FileRequired, FileField
from wtforms import TextField
from wtforms.validators import DataRequired


class UploadForm(FlaskForm):
    batch = TextField(
            'batch',
            validators=[
                    DataRequired(),
                    ]
            )
    
    cell = TextField(
            'cell',
            validators=[
                    DataRequired(),
                    ]
            )
    
    fs = TextField(
            'fs',
            validators=[
                    DataRequired(),
                    ],
            default='200'
            )
    
    stimfmt = TextField(
            'stim format',
            validators=[
                    DataRequired(),
                    ],
            default='ozgf'
            )
    
    chancount = TextField(
            'channel count',
            validators=[
                    DataRequired(),
                    ],
            default='18'
            )
    
    upload = FileField(
            'file',
            validators=[
                    FileRequired(),
                    FileAllowed(['.pkl', '.mat'],
                                'File must be .pkl or .mat'
                                ),
                    ]
            )