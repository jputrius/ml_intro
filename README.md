# Introduction
This repository hosts source files for building the presentation I use to give an introductory course to machine learning at Vilnius University.

This course was previously run by Rokas Tamošiūnas.
You can find the lecture notes for that course [here](https://github.com/trokas/ai_primer).
These present notes have been adapted from those.

The course slides are available [here](https://jputrius.github.io/ml_intro/).

## Running the notebooks

The easiest way to run the notebooks is to open them in Google Colab (wou will need a google account).
The Colab links for the notebooks can be found [here](https://jputrius.github.io/ml_intro/).

However, if your machine has a GPU then I recommend setting up the project locally.
Here is a step by step guide:

1. Clone this repository.
2. Download and install [poetry](https://python-poetry.org/). Poetry is a tool for managing python packages and  environments.
3. Make sure you have python version `3.10`, `3.11` or `3.12` installed on your machine. In windows you can download and install extra python versions from [here](https://www.python.org/downloads/). On Linux or Mac I recommend to use [pyenv](https://github.com/pyenv/pyenv) to manage extra python versions.
4. The notebooks use PyTorch for creating and training neural networks. You will need a PyTorch version that is compatible with your hardware. To check the version go to PyTorch's [website](https://pytorch.org/). Once you figure out the version, you will need to edit the `pyproject.toml` file to specify it. There is a comment saying which line you need to modify.
5. Run `poetry install`.
6. You are good to go! To run the notebooks you can:
    1. Use `jupyterlab`. Run `poetry run jupyter-lab` to open it (it should open in your default browser).
    2. Use [VS Code](https://code.visualstudio.com/). Make sure you select the python kernel that was created by poetry. It should be in the `./.env` directory.