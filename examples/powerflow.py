from ldrestoration.dssparser.dssparser import DSSManager   
from ldrestoration.core.restorationmodel import RestorationModel
from ldrestoration.core.dataloader import load_data

# this will not be present in the future version
from pyomo.environ import value

# restoration model
system_data = load_data("parsed_data_123bus")
rm = RestorationModel(system_data, faults=None)

# load constraints
rm.constraints_base()

# load objective
rm.objective_load_only()

# activate or deactivate constraints
# restore.model.voltage_balance.deactivate()

# solve the model
model = rm.solve_model(solver='gurobi',tee=True)


# observe results
print(f"Substation flow: Pa = {value(model.Pija[0])}, Pb = {value(model.Pijb[0])}, Pc = {value(model.Pijc[0])}")

for edges in model.sec_test:  
    print(f"connectivity status for sectionalizers xij[{edges}] = {value(model.xij[edges])}")
  
for edges in model.tie_test:  
    print(f"connectivity status for tie switches {model.edges[edges]}, xij[{edges}] = {value(model.xij[edges])}")