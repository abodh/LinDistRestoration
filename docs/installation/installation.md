# Installation Overview

It is *highly recommended* for the users to create a separate conda enviroment before proceeding further. LinDistRestoration requires a python version of 3.10 or greater. Please follow the steps below to create a conda environment. Please follow the instructions <a href="https://www.anaconda.com/download" target="_blank">here</a> to install the latest version of Anaconda. 


## Creating a python environment
If you are in this step, we assume that you have successfully installed Anaconda in your system. While the steps here are specific to Windows users, Linux or Mac users can follow a similar steps accordingly.

`WARNING: LinDistRestoration has not been tested on Linux or Mac as of now.`

Here are the steps to create a new conda environment:

* Open Anaconda Prompt from the start menu. The `(base)` environment should be currently active. It is advised not to use the base environment for any package installations.
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