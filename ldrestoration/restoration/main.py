from __future__ import annotations

from ldrestoration import (DataLoader, 
                           RestorationModel)

def restoration_base(data_path: str, 
                     faults: list[tuple] = None,
                     base_kV_LL: float = None,
                     vmax: float = 1.05,
                     vmin: float = 0.95,
                     vsub: float = 1.05,
                     psub_max: float = 5000,
                     ) -> RestorationModel:
    
    """Base restoration model to maximize load restoration with base constraints

    Args:
        data_path (str): path of the required data files for the restoration module
        faults (list[str]): list of line element edges with faults in (u,v) format
        base_kV_LL (float): basekV of the circuit.
        vmax (float, optional): Maximum voltage of the system in per unit (pu). Defaults to 1.05.
        vmin (float, optional): Minimum voltage of the system in per unit (pu). Defaults to 0.95.
        vsub (float, optional): reference substation voltage of the system in per unit (pu). Defaults to 1.05.
        psub_max (float, optional): maximum substation flow of the system in kW. Defaults to 5000 kW.
    Returns:
        RestorationModel: restoration model object
    """    
  
    # instantiate the data loader object and get the required data for the restoration
    data_object = DataLoader(data_path)
    
    # instantiate the restoration object. Provide faulted edges in (u,v) format, if available.
    restoration_object = RestorationModel(data_object, faults=faults)
    
    # load the constraint sets -> base constraints set contains all the necessary constraints for restoration
    # provide necessary voltage limit values and substation reference voltage
    restoration_object.constraints_base(base_kV_LL=base_kV_LL,
                                        vmax=vmax, 
                                        vmin=vmin, 
                                        vsub=vsub,
                                        psub_max=psub_max)    
    return restoration_object







