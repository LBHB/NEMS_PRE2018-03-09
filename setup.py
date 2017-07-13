import codecs
import os.path
from setuptools import find_packages, setup

NAME = 'NEMS'

version = 'pre-alpha'

with codecs.open('README.md', encoding='utf-8') as f:
    long_description = f.read()

GENERAL_REQUIRES = ['numpy', 'scipy', 'matplotlib', 'flask', 'sqlalchemy']
WEB_REQUIRES = ['flask', 'mpld3', 'bokeh', 'flask-socketio', 'eventlet']
DB_REQUIRES = ['sqlalchemy', 'pymysql']

setup(
    name=NAME,
    version=version,
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    zip_safe=True,
    author='LBHB',
    author_email='lbhb.ohsu@gmail.com',
    description='Neural encoding model system',
    long_description=long_description,
    url='http://neuralprediction.org',
    install_requires=GENERAL_REQUIRES,
    extras_require={
        'web': WEB_REQUIRES + DB_REQUIRES,
        'database': DB_REQUIRES,
    },
    #setup_requires=['pytest-runner'],
    #tests_require=['pytest'],
    #license='MIT',
    classifiers=[]
)
