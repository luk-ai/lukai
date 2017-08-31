from setuptools import setup, find_packages
import os

packages = find_packages('.')

packages += ['lukai.proto.{}'.format(pkg) for pkg in next(os.walk('lukai/proto'))[1]]

setup(
    name='lukai',
    packages=packages,
    version='0.1',
    description='Luk.ai management library',
    install_requires=['grpcio', 'six'],
    author='Tristan Rice',
    author_email='rice@fn.lc',
    url='https://github.com/luk-ai/lukai/tree/master/py',
    keywords=['ml','machine learning','lukai'],
    classifiers=[],
)
