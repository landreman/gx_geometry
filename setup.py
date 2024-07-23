from setuptools import setup, find_packages

def parse_requirements(filename):
    """ Load requirements from a pip requirements file """
    with open(filename, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines if line and not line.startswith('#')]

requirements = parse_requirements('requirements.txt')

setup(
    name="gx_geometry",
    version="0.0.1",
    url="https://github.com/landreman/gx_geometry.git",
    author="Matt Landreman",
    author_email="mattland@umd.edu",
    # description='Description',
    packages=find_packages(),
    entry_points = {
        'console_scripts': ['gx_geometry=gx_geometry.module:run_module'],
    },
    install_requires=requirements,
)
