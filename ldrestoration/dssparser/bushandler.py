from types import ModuleType
from typing import Union
import logging

from ldrestoration.utils.decors import timethis
from ldrestoration.utils.loggerconfig import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

class BusHandler:
    def __init__(self, 
                 dss_instance: ModuleType) -> None:
        """Initialize a BusHandler instance. This instance deals with bus (node) related data from the distribution model.
        Note: Bus and Nodes are two different concepts in distribution systems modeling and are used interchangably here 
        for simplicity.
        
        Args:
            dss_instance (ModuleType): redirected opendssdirect instance 
        """
        
        self.dss_instance = dss_instance 
    
    @timethis
    def get_buses(self) -> list[dict[str,Union[int,str,float]]]: 
        """Extract the bus data -> name, basekV, latitude, longitude from the distribution model.

        Returns:
            bus_data (list[dict[str,Union[int,str,float]]]): list of bus data for each buses
        """
        all_buses_names = self.dss_instance.Circuit.AllBusNames()        
        bus_data = []        
        for bus in all_buses_names:
            
            # need to set the nodes active before extracting their info 
            self.dss_instance.Circuit.SetActiveBus(bus)
                        
            # be careful that X gives you lon and Y gives you lat
            bus_coordinates = dict(name = self.dss_instance.Bus.Name(),
                                   basekV = round(self.dss_instance.Bus.kVBase(),2),
                                   latitude = self.dss_instance.Bus.Y(),
                                   longitude = self.dss_instance.Bus.X())
            
            bus_data.append(bus_coordinates)
        return bus_data