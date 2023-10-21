from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='gretta',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'numba',
    ],
    author='Evgeny Frolov',
    description='Tensor-based SSA for sparse datasets with spatiotemporal information',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/recspert/gretta',
)