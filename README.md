# Introduction
This repository hosts source files for building the presentation I use to give an introductory course to data science and machine learning at Vilnius University.

Most of the material here is based on Rokas Tamošiūnas' [lecture notes](https://github.com/trokas/ai_primer).

## Running the notebooks

This project uses [poetry](https://python-poetry.org/) to manage python packages and environments.

With poetry installed, run `poetry install`. This will create a python environment in the `.env` folder and install the necessary packages. Afterwards you can run the notebooks in the `chapters` folder. Remember to select the python kernel from the `.env` folder!

Note that the `pyproject.toml` file is set up to install PyTorch without CUDA. If your computer has a CUDA enabled graphics card you will need to modify it a bit. In the `pyproject.toml` file change the url in the `[[tool.poetry.source]]` section to the one appropriate for your CUDA version. You can find the urls for different CUDA versions [here](https://pytorch.org/get-started/locally/).