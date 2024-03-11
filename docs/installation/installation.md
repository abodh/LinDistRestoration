# Installation Overview

Before installing *ldrestoration*, it is *highly recommended* for the users to create a separate conda enviroment before installing the package. *ldrestoration* requires a python version of 3.10 or greater. Please follow the steps below to create a conda environment. Please follow the instructions <a href="https://www.anaconda.com/download" target="_blank">here</a> to install the latest version of Anaconda. 


# Creating a python environment
In this step, we assume that you have successfully installed Anaconda in your system. Here are the steps to create a new conda environment for *ldrestoration*. We provide a basic guidiline for Windows while Linux or Mac users can adapt accordingly.

* Open Anaconda Prompt from the start menu. Initially, the `(base)` environment should be currently active. It is advised not to use the base environment for any package installations.
* create a new conda evironment using the following command:
```bash
conda create -n ldrestoration python=3.10
```

This command will create a new environment, `ldrestoration`, with the specific version of python i.e., `3.10.x`. Here, the latest version of python 3.10 will be installed. A specific version can be installed with double equal sign i.e., `python==3.10.2` for example.

* activate the environment:
The following command will then activate the newly created environment
```bash
conda activate ldrestoration
```
You can then install necessary libraries in this environment to maintain consistency within the same package and avoid any version conflict of the dependent packages with other installations.


# Installing LinDistRestoration
Please review different installation methods shown in the left tab *after* you have successfully created an environment for installing the package. Currently, we only support installation via GitHub and will soon release the first version to make the installation possible via pip. 

# Installing solvers:
We have also provided installation guides for different opensource or commercial solvers. You may choose to install and use other solvers as well. The current version of ldrestoration is tested on the following solvers:

- Gurobi 10.0.0