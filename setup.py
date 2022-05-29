from setuptools import setup, find_packages

setup(
    author="kamiteku557",
    name="my_deep_learning_from_scrach_4",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)