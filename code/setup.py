from setuptools import setup, find_packages

setup(
    author="kamiteku557",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)