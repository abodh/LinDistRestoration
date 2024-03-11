from types import ModuleType
from typing import Union

import logging
from ldrestoration.utils.loggerconfig import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class TransformerHandler:
    """TransformerHandler extracts the transformers (step down, step up, or split-phase service transformers) in the distribution model.
    
    Args:
        dss_instance (ModuleType): redirected opendssdirect instance 
        
    To do:
        * Address the extraction of delta connected primary in split-phase transformers
    """
    def __init__(self, 
                 dss_instance: ModuleType) -> None:
        """Initialize a TransformerHandler instance. This instance deals with transformers in the distribution model.
        
        Args:
            dss_instance (ModuleType): redirected opendssdirect instance 
        """
        
        self.dss_instance = dss_instance 
    
    def get_splitphase_primary(self) -> dict[str,str]:    
        """Gets the primary phase information from split phase transformers to refer all loads to the primary

        Returns:
            splitphase_node_primary (dict[str,str]): A dictionary with secondary node as key and associated phase in primary as value
            for eg. for ['A.3', 'B.1.0', 'B.0.2'] this will return {'B':['3']}
        """   
        splitphase_node_primary = {}                
        transformer_flag = self.dss_instance.Transformers.First()        
        while transformer_flag:
            
            if (self.dss_instance.CktElement.NumPhases() != 3) and self.dss_instance.Transformers.NumWindings() == 3:
                # a split phase transformer is a three winding single phase transformer (two phase primary accounts for delta)

                # name extracted of the secondary and phase extracted of the primary
                bus_name = self.dss_instance.CktElement.BusNames()[1].split('.')[0]
                bus_phases = self.dss_instance.CktElement.BusNames()[0].split('.')[1:]
                
                if bus_name not in splitphase_node_primary:
                    splitphase_node_primary[bus_name] = bus_phases

            transformer_flag = self.dss_instance.Transformers.Next()
            
        return splitphase_node_primary
    
    def get_transformers(self) -> list[dict[str,Union[int,str,float]]]: 
        """Extract the bus data -> name, basekV, latitude, longitude from the distribution model.

        Returns:
            bus_data (list[dict[str,Union[int,str,float]]]): list of bus data for each buses
        """
        transformer_flag = self.dss_instance.Transformers.First()      
        transformer_data = []  
        while transformer_flag:
            each_transformer = dict(name = self.dss_instance.Transformers.Name(),
                                    numwindings = self.dss_instance.Transformers.NumWindings(),
                                    connected_from = self.dss_instance.CktElement.BusNames()[0].split('.')[0],
                                    connected_to = self.dss_instance.CktElement.BusNames()[1].split('.')[0])
            
            transformer_data.append(each_transformer)
            transformer_flag = self.dss_instance.Transformers.Next() 
        return transformer_data   