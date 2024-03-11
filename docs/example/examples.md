# Examples

## **Parse OpenDSS data**

Parse an OpenDSS model and save it in a local directory. An example script to parse the OpenDSS master file can be accessed from `examples/dataparser/data_parser.py`

```python
from ldrestoration import DSSManager  

# 1. instantiate the DSSManager with the OpenDSS master file
dss_object = DSSManager(r"../test_cases/ieee123Bus/Run_IEEE123Bus.dss")

# 2. Parse the dss data to the ldrestoration data structure
dss_object.parsedss()

# 3. Save the parsed data in the desired output folder  
dss_object.saveparseddss(folder_name="parsed_data_ieee123")

```

## **Instantiate restoration model**

Instantiate a restoration model by loading the data into the model.

```python
from ldrestoration import RestorationBase

# 1. identify the path of the parsed data files
parsed_data_path = "parsed_data_iee123/"

# 2. instantiate the restoration model
rm = RestorationBase(parsed_data_path, base_kV_LL=4.16)
```

## **Solve the restoration model**

solve the restoration model. To solve the model it needs to be instantiated first as shown above.

```python
# 3. identify objective function
rm.objective_load_only()

# 4. solve the restoration model with the solver of your choice 
rm_solved, results = rm.solve_model(solver='glpk')
```

## **Extract results and plot**

Once the model is solved, it can be plotted on the map to observe the restoration policy visually. To save the results in a csv file, you can pass `save_results=True` and `results_filename` as 

```python
rm.solve_model(solver='glpk', save_results=True, results_filename="ieee123solved.csv")
```

The solved restoration model can be plotted on a geographical map using the following snippet 

```python
from ldrestoration.utils.plotnetwork import plot_solution_map

# plot the restoration solution. It requires the solved pyomo model, networkx tree, and
# networkx graph. The latter two are embedded within the pyomo model for easier access.
plot_solution_map(rm_solved, rm.network_tree, rm.network_graph)
```