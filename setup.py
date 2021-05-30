from setuptools import setup
from pip.req import parse_requirements

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements('requirements.txt')

# reqs is a list of requirement
reqs = [str(ir.req) for ir in install_reqs]

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='Measurements',
    url='https://github.com/skrzypczykt/CycleGAN',
    author='Tomasz Skrzypczyk',
    author_email='skrzypczykt@gmail.com',
    # Needed to actually package something
    packages=['gans'],
    # Needed for dependencies
    install_requires= reqs,
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='An example of a python package from pre-existing code and existing gan implementation',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)