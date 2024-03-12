# Overview

Before diving into this section, please ensure that you have:

1. successfully installed the *ldrestoration* package in a new environment and have activated it.
2. Have installed a solver (open source or commercial).
3. have either a working OpenDSS model or CSV input data files compatible with *ldrestoration*.  

For 1 and 2, please refer to the [installation](../installation/installation.md){:target="_blank"} page for more details.

The general structure to solve a restoration model, assuming we are starting from the OpenDSS model, is to parse and save the required data structure, formulate the restoration model, solve the restoration model, and create plots or access results for further analysis. The examples on each of these processes can be accessed from [examples](examples.md){:target="_blank"}.

Assuming the data is parsed from OpenDSS in the current directory, the optimal power flow for IEEE123 bus system with base model can be executed with the following steps. Here, `rm_solved` is the solved *pyomo* restoration model and `results` is the *pyomo* results object. 

```python
from ldrestoration import RestorationBase

# 1. identify the path of the parsed data files
parsed_data_path = "parsed_data_iee123/"

# 2. instantiate the restoration model
rm = RestorationBase(parsed_data_path, base_kV_LL=4.16)

# 3. identify objective function
rm.objective_load_only()

# 4. solve the restoration model with the solver of your choice 
rm_solved, results = rm.solve_model(solver='glpk')
```

