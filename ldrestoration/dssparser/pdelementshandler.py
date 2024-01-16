from types import ModuleType
from typing import Union
import numpy.typing as npt
import numpy as np

import logging
from ldrestoration.utils.logger_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)



class PDElementHandler:
    def __init__(self, 
                 dss_instance: ModuleType) -> None:
        """Initialize a PDElementHandler instance. This instance deals with all the power delivery elements -> lines, transformers,
        reactors, and capacitors. ALthough we have separate handlers for a few of them, we extract the PDelements here as they represent 
        edges for out network  
        
        Args:
            dss_instance (ModuleType): redirected opendssdirect instance
        """
        
        self.dss_instance = dss_instance 
    
    def __get_zmatrix(self) -> npt.NDArray[np.complex128]:
        """Returns the z_matrix of a specified pdelement. 

        Returns:
            z_matrix (npt.NDArray[np.complex128]): 3x3 numpy complex array of the z_matrix corresponding to the each of the phases
        """
        
        if ((len(self.dss_instance.CktElement.BusNames()[0].split('.')) == 4) or 
            (len(self.dss_instance.CktElement.BusNames()[0].split('.')) == 1)):
            
            # this is the condition check for three phase since three phase is either represented by bus_name.1.2.3 or bus_name    
            z_matrix = np.array(self.dss_instance.Lines.RMatrix()) + 1j * np.array(self.dss_instance.Lines.XMatrix())
            z_matrix = z_matrix.reshape(3, 3)
            
            return z_matrix
        
        else:
            
            # for other than 3 phases            
            active_phases = [int(phase) for phase in self.dss_instance.CktElement.BusNames()[0].split('.')[1:]]
            z_matrix = np.zeros((3, 3), dtype=complex)
            r_matrix = self.dss_instance.Lines.RMatrix()
            x_matrix = self.dss_instance.Lines.XMatrix()
            counter = 0
            for _, row in enumerate(active_phases):
                for _, col in enumerate(active_phases):
                    z_matrix[row - 1, col - 1] = complex(r_matrix[counter], x_matrix[counter])
                    counter = counter + 1

            return z_matrix
        
    def get_pdelements(self) -> list[dict[str,Union[int,str,float, npt.NDArray[np.complex128]]]]:
        
        """Extract the list of PDElement from the distribution model. Capacitors are excluded.

        Returns:
            pdelement_data (list[dict[str,Union[int,str,float, npt.NDArray[np.complex128]]]]): 
            list of pdelements with required information
        """
             
        element_activity_status = self.dss_instance.PDElements.First()
        pdelement_data = []

        while element_activity_status:
            element_type = self.dss_instance.CktElement.Name().lower().split('.')[0] 
            
            # capacitor is a shunt element  and is not included
            if element_type != 'capacitor':                
                #"Capacitors are shunt elements and are not modeled in this work. Regulators are not modeled as well."
                
                if element_type == 'line':
                    each_element_data = {
                        'name': self.dss_instance.Lines.Name(),
                        'element': element_type,
                        # from opendss manual -> length units = {none|mi|kft|km|m|ft|in|cm}
                        'length_unit': self.dss_instance.Lines.Units(),
                        'z_matrix': self.__get_zmatrix(),
                        'length': self.dss_instance.Lines.Length(),
                        'from_bus': self.dss_instance.Lines.Bus1().split('.')[0],
                        'to_bus': self.dss_instance.Lines.Bus2().split('.')[0],
                        'num_phases': self.dss_instance.Lines.Phases(),
                        'is_switch': self.dss_instance.Lines.IsSwitch(),
                        'is_open': (self.dss_instance.CktElement.IsOpen(1, 0) or self.dss_instance.CktElement.IsOpen(2, 0))
                    }      
                
                else:
                    # everything other than lines but not capacitors i.e. transformers, reactors etc.
                    # The impedance matrix for transformers and reactors are modeled as a shorted line here. 
                    # Need to work on this for future cases and replace with their zero sequence impedance may be
                    
                    each_element_data = {
                    'name': self.dss_instance.CktElement.Name().split('.')[1],
                    'element': element_type,
                    # from opendss manual -> length units = {none|mi|kft|km|m|ft|in|cm}
                    'length_unit': 2,                
                    'z_matrix': np.zeros((3, 3)) + 1j * np.zeros((3, 3)),
                    'length': 0.001,
                    'from_bus': self.dss_instance.CktElement.BusNames()[0].split('.')[0],
                    'to_bus': self.dss_instance.CktElement.BusNames()[1].split('.')[0],
                    'num_phases': self.dss_instance.CktElement.NumPhases(),
                    'is_switch': False,
                    'is_open': False
                    }
                    
                pdelement_data.append(each_element_data)
            element_activity_status = self.dss_instance.PDElements.Next()
            
        return pdelement_data