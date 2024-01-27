from types import ModuleType
from typing import Union
import numpy as np

import logging
from ldrestoration.utils.loggerconfig import setup_logging

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
    
    def __get_line_zmatrix(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the z_matrix of a specified line element. 

        Returns:
            real z_matrix, imag z_matrix (np.ndarray, np.ndarray): 3x3 numpy array of the z_matrix corresponding to the each of the phases(real,imag)
        """
        
        if ((len(self.dss_instance.CktElement.BusNames()[0].split('.')) == 4) or 
            (len(self.dss_instance.CktElement.BusNames()[0].split('.')) == 1)):
            
            # this is the condition check for three phase since three phase is either represented by bus_name.1.2.3 or bus_name    
            z_matrix = np.array(self.dss_instance.Lines.RMatrix()) + 1j * np.array(self.dss_instance.Lines.XMatrix())
            z_matrix = z_matrix.reshape(3, 3)
            
            return np.real(z_matrix), np.imag(z_matrix)
        
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

            return np.real(z_matrix), np.imag(z_matrix)
    
    def __get_nonline_zmatrix(self) -> list[list[float]]:
        """Returns the z_matrix of a specified element other than the line element. 

        Returns:
            z_matrix (list[list[float]]): list of list of float of z matrices (same fo real and imag)
        """        
        # hash map for the element z matrices other than lines
        # this is temporary and should be fixed later to replace with the actual impedances of the element.
        elements_z_matrix = {
            ('1',): [[0.001,0,0],[0,0,0],[0,0,0]],
            ('2',): [[0,0,0],[0,0.001,0],[0,0,0]],
            ('3',): [[0,0,0],[0,0,0],[0,0,0.001]],
            ('1','2'): [[0.001,0,0],[0,0.001,0],[0,0,0]],
            ('2','3'): [[0,0,0],[0,0.001,0],[0,0,0.001]],
            ('1','3'): [[0.001,0,0],[0,0,0],[0,0,0.001]],
            ('1','2','3'): [[0.001,0,0],[0,0.001,0],[0,0,0.001]]                       
        }        
            
        if self.dss_instance.CktElement.NumPhases() == 3:
            return elements_z_matrix[('1','2','3')]
        else:
            bus_phases = self.dss_instance.CktElement.BusNames()[0].split('.')[1:]
            return elements_z_matrix[tuple(bus_phases)]
        
    def element_phase_identification(self, 
                                     element_phases=list[str]) -> list[str]:
        """Match the phase from the number convention to letter convention i.e. 123 -> abc

        Returns:
            set[str]: set of phases converted to letter type
        """        
        # create a dict of mapper
        phasemap = {
            '1': 'a',
            '2': 'b',
            '3': 'c'
        }     
           
        # identify and return corresponding dss phases in numbers to the letters
        return {phasemap[dss_phase] for dss_phase in element_phases}   
    
        
    def get_pdelements(self) -> list[dict[str,Union[int,str,float, np.ndarray]]]:
        
        """Extract the list of PDElement from the distribution model. Capacitors are excluded.

        Returns:
            pdelement_list (list[dict[str,Union[int,str,float, np.ndarray]]]): 
            list of pdelements with required information
        """        
             
        element_activity_status = self.dss_instance.PDElements.First()
        pdelement_list = []

        while element_activity_status:
            element_type = self.dss_instance.CktElement.Name().lower().split('.')[0] 
            
            # capacitor is a shunt element  and is not included
            if element_type != 'capacitor':                
                #"Capacitors are shunt elements and are not modeled in this work. Regulators are not modeled as well."
                if element_type == 'line':
                    z_matrix_real, z_matrix_imag = self.__get_line_zmatrix()
                    each_element_data = {
                        'name': self.dss_instance.Lines.Name(),
                        'element': element_type,
                        # from opendss manual -> length units = {none|mi|kft|km|m|ft|in|cm}
                        'length_unit': self.dss_instance.Lines.Units(),
                        'z_matrix_real': z_matrix_real.tolist(),
                        'z_matrix_imag': z_matrix_imag.tolist(),
                        'length': self.dss_instance.Lines.Length(),
                        'from_bus': self.dss_instance.Lines.Bus1().split('.')[0],
                        'to_bus': self.dss_instance.Lines.Bus2().split('.')[0],
                        'num_phases': self.dss_instance.Lines.Phases(),
                        'phases':{'a','b','c'} if self.dss_instance.CktElement.NumPhases() == 3 else
                        self.element_phase_identification(element_phases=self.dss_instance.CktElement.BusNames()[0].split('.')[1:]),
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
                    'length_unit': 0,                
                    'z_matrix_real': self.__get_nonline_zmatrix(),
                    'z_matrix_imag': self.__get_nonline_zmatrix(),
                    'length': 0.001,
                    'from_bus': self.dss_instance.CktElement.BusNames()[0].split('.')[0],
                    'to_bus': self.dss_instance.CktElement.BusNames()[1].split('.')[0],
                    # for non lines dss.Lines does not work so we need to work around with CktElement
                    # CktElement is activated along with PDElements
                    'num_phases': self.dss_instance.CktElement.NumPhases(),
                    'phases':{'a','b','c'} if self.dss_instance.CktElement.NumPhases() == 3 else
                    self.element_phase_identification(element_phases=self.dss_instance.CktElement.BusNames()[0].split('.')[1:]),
                    'is_switch': False,
                    'is_open': False
                    }
                    
                pdelement_list.append(each_element_data)
            element_activity_status = self.dss_instance.PDElements.Next()

        return pdelement_list