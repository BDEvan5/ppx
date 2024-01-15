from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'PPO using JAX'
LONG_DESCRIPTION = 'JAX implementations for PPO algorithms'

# Setting up
setup(
        name="ppx", 
        version=VERSION,
        author="Benjamin Evans",
        author_email="<benjaminevans316@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        
        keywords=['python'],
        classifiers= [
            "Programming Language :: Python :: 3.9",
        ]
)
