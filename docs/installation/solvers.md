# Installation of solvers

You can either use open source or commercial solvers. The current version of LinDistRestoration is tested on the following solvers:

- Gurobi 10.0.0
- *TBD*

The optimization model in LinDistRestoration is based off of *Pyomo*, which requires solvers to solve the model. For commercial solver, we recommend using *Gurobi* and if you are an academic researcher, you can leverage their free academic license. For more information please visit their webpage <a href="https://www.gurobi.com/academia/academic-program-and-licenses/" target="_blank">here</a>.

# open source solvers:
At this point, we have not tested any open source solvers and aim to do that in the next release. However, we believe that you can install the following solvers to get started if you do not have access to commercial solver.

* **GLPK**:
GLPK (GNU Linear Programming Kit) is an open source solver for linear programming, mixed-integer programming and other related problems. To install `glpk`, please activate the conda evnironment using `conda activate ldrestoration` and type the following command:

```shell
conda install -c conda-forge glpk
```

* **CBC**:
Cbc (Coin-or branch and cut) is an open source solver specially for mixed integer linear programming written in C++. To install `cbc`, please activate the conda evnironment using `conda activate ldrestoration` and type the following command:

```shell
conda install -c conda-forge coin-or-cbc
```

* **HiGHS**:
HiGHS is a high performance serial and parallel solver, which is popular for sparse linear, mixed-integer, and quadratic programming problems. 
```shell
conda install conda-forge::highs
```