from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'PPO using JAX'
LONG_DESCRIPTION = 'JAX implementations for PPO algorithms'

# Setting up
setup(
        name="ppox", 
        version=VERSION,
        author="Benjamin Evans",
        author_email="<bdevans@sun.ac.za>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python'],
        classifiers= [
            "Programming Language :: Python :: 3.9",
        ]
)
