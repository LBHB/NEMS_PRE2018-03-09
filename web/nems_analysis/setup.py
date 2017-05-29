from setuptools import setup

setup(
    name='nems_analysis',
    packages=['nems_analysis'],
    include_package_data=True,
    install_requires=[
            'flask',
            'pandas',
            'bokeh',
    ],
)