from setuptools import find_packages, setup

setup(
    name="spine",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    version="0.1.0",
    description="Spine estimation with NNs",
    author="Nicholas Santavas",
    license="",
)
