from ldrestoration import RestorationBase, DataLoader

"""Please follow the documentation for a quick start. The documentation will be updated soon to 
reflect this example and explanation of additional functionalities in LinDistRestoration. 
"""


# instantiate the data loader object and get the required data for the restoration from csv
data_object = DataLoader(data_folder_path=r"../dataparser/parsed_data_9500_noder")
data_model = data_object.from_csv()

# fault in line 35 - 36
rm = RestorationBase(
    data_model=data_model,
    faults=[("d5534967-1_int", "m1089174")],
    base_kV_LL=12.47,
    vmin=0.9,
    vmax=1.1,
    psub_max=5000,
)

# load the objective of the problem
rm.objective_load_and_switching()

# solve the restoration model
restoration_model, results = rm.solve_model(solver="appsi_highs", tee=True)
rm.save_variable_results(results)
