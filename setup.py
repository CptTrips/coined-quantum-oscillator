from setuptools import setup, find_packages


setup(
    name = "Coined Quantum Oscillator",
    version = "0.1.0",
    packages = find_packages(exclude=["*.test", "test", "*.test.*"]),
    install_requires = ['argparse', 'numpy', 'matplotlib']
)
