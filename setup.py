from setuptools import setup, find_packages

setup(
    name='halton',
    version='0.1.0',
    author='Ohad Rubin',
    author_email='your.email@example.com',
    description='A short description of your library',
    packages=["halton"],
    install_requires=[
        'numpy',
        'absl-py',
        "click"
    ],
)
