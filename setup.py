import codecs
import os.path
from setuptools import find_packages, setup

NAME = 'NEMS'

version = 'pre-alpha'

with codecs.open('README.md', encoding='utf-8') as f:
    long_description = f.read()

GENERAL_REQUIRES = [
        'numpy', 'scipy', 'matplotlib', 'sqlalchemy', 'pymysql',
        'mpld3', 'boto3', 'tensorflow',
        ]

setup(
    name=NAME,
    version=version,
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    zip_safe=True,
    author='LBHB',
    author_email='lbhb.ohsu@gmail.com',
    description='Neural Encoding Model System',
    long_description=long_description,
    url='http://neuralprediction.org',
    install_requires=GENERAL_REQUIRES,
    classifiers=[]
)
