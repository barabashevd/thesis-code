from setuptools import setup, find_packages
requires = open("requirements.txt", "r").read().strip().split()

setup(
    name="Comparing Methods for Detecting Concept Drift in Data Streams",
    version="0.1",
    author="Daniil Barabasev",
    author_email="barbda1@student.cvut.cz",
    description="Supporting code for my thesis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/barabashevd/thesis-code",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
    python_requires='>=3.9',
    install_requires=requires
)