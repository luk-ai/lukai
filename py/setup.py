from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='lukai',
    packages=find_packages('.'),
    version='0.4',
    description='Luk.ai management library',
    install_requires=['grpcio', 'six', 'srvlookup'],
    author='Tristan Rice',
    author_email='rice@fn.lc',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/luk-ai/lukai/tree/master/py',
    keywords=['ml','machine learning','lukai'],
    classifiers=(
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
