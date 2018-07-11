from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='lukai',
    packages=find_packages('.'),
    version='0.6',
    description='Luk.ai management library',
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    install_requires=['grpcio', 'six', 'srvlookup', 'tensorflow==1.8.0', 'tensorflowjs', 'protobuf'],
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
