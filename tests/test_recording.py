import nems.epoch as ep
from nems.recording import Recording


def test_recording_loading():
    '''
    Test the loading and saving of files to various HTTP/S3/File routes.
    '''
    # Local filesystem
    #rec0 = Recording.load("/home/ivar/git/nems/signals/TAR010c-18-1.tar.gz")
    #rec1 = Recording.load("/auto/data/tmp/recordings/TAR010c-18-1.tar.gz")
    #rec2 = Recording.load("file:///auto/data/tmp/recordings/TAR010c-18-1.tar.gz")

    # HTTP
    #rec3 = Recording.load("http://potoroo:3001/recordings/TAR010c-18-1.tar.gz")
    #rec4 = Recording.load("http://potoroo/baphy/271/TAR010c-18-1")

    # S3
    # Direct access (would need AWS CLI lib? Maybe not best idea!)
    # rec5 = Recording.load('s3://mybucket/myfile.tar.gz')
    # Indirect access via http:
    rec6 = Recording.load('https://s3-us-west-2.amazonaws.com/nemspublic/sample_data/TAR010c-18-1.tar.gz')
