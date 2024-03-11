from ldrestoration import RestorationBase
from ldrestoration.utils.plotnetwork import plot_solution_map

"""This is a bit more advanced example than the one in the documentation. 
Please follow the documentation for a quick start. The documentation will be updated soon to 
reflect this example and explanation of additional functionalities in ldrestoration. 
"""


# no fault case
# rm = RestorationModel(data_object)

# fault in ln5593236-6
# rm = RestorationModel(data_object, faults=[('m1047515', 'm1047513')]) 

# fault in ln6198013-1
# rm = RestorationModel(data_object, faults=[('m1026331', 'm1026330')]) 

# fault in LN5593236-6, LN6291253-1, LN5714974-1, LN5799561-1
rm = RestorationBase('../dataparser/parsed_data_9500_der',
                     faults=[('m1047515', 'm1047513'),
                             ('m1108315', 'm1108311'),
                             ('m1209811', 'm1209807'),
                             ('m1142843', 'l3081380')],
                     base_kV_LL=12.47,
                     vmin=0.93,
                     vmax=1.07,
                     psub_max=5000)

# load the objective of the problem
# the pyomo model can be accessed via rm9500.model
rm.objective_load_switching_and_der()


# the constraints can be accessed as following
# rm._constraints_list

# the name of each of the constraints can be obtained as following
# rm._constraints_list[0].name 

# constraints can be deactivated as following
# rm.model.voltage_balance.deactivate()
# rm.model.substation_positive_flow.deactivate()

# solve the restoration model
# after solving the restoration model is returned back
restoration_model, results = rm.solve_model(solver = 'gurobi',
                                            tee = True)

# network plot
# plot_solution(restoration_model, rm.network_tree, rm.network_graph)
