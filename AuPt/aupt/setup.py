"""setup.py for the project"""
from setuptools import setup, find_packages

setup(
    name="aupt",
    version="0.0",
    author="Sebastian Franco Ulloa",
    author_email="sebastian@sebastianfu.com",
    description="Package with analysis tools for molecular dynamics simulations",
    python_requires=">=3.7.0",
    packages=find_packages('src'),
    package_dir={'aupt': 'src/aupt'},
    include_package_data=True,
    install_requires=[
        "setuptools>=67.6.0",
        "MDAnalysis>=2.4.2",
        "numpy>=1.24.2",
        "matplotlib>=3.7.1"
    ]
)
