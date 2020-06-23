from os.path import join, dirname, realpath
from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, (
    "The Fired Up repo is designed to work with Python 3.6 and greater."
    + "Please install it before proceeding."
)

with open(join("fireup", "version.py")) as version_file:
    exec(version_file.read())

setup(
    name="fireup",
    py_modules=["fireup"],
    version=__version__,
    install_requires=[
        "cloudpickle",
        "gym[atari,box2d,classic_control]",
        "ipython",
        "joblib",
        "matplotlib",
        "mpi4py",
        "numpy",
        "pandas",
        "pytest",
        "psutil",
        "scipy",
        "seaborn",
        "torch>=1.5.1",
        "tqdm",
        "wandb",
    ],
    description="PyTorch clone of OpenAI's Spinning Up which is a teaching tools for introducing people to deep RL.",
    author="Kashif Rasul, Joshua Achiam",
    license="MIT",
)
