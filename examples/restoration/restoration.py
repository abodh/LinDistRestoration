from ldrestoration import RestorationBase

"""Please follow the documentation for a quick start. The documentation will be updated soon to 
reflect this example and explanation of additional functionalities in LinDistRestoration. 
"""

# fault in line 35 - 36
rm = RestorationBase('../dataparser/parsed_data_ieee123',
                     faults=[('35', '36'),
                             ('m1108315', 'm1108311'),
                             ('m1209811', 'm1209807'),
                             ('m1142843', 'l3081380')],
                     base_kV_LL=4.16,
                     vmin=0.95,
                     vmax=1.05,
                     psub_max=5000)

# load the objective of the problem
rm.objective_load_and_switching()

# solve the restoration model
restoration_model, results = rm.solve_model(solver = 'gurobi',
                                            tee = True)