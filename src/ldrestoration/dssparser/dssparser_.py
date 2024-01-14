from __future__ import annotations
from typing import Union, Optional
import numpy.typing as npt
from types import ModuleType

import opendssdirect as dss 
import pandas as pd
import numpy as np
import networkx as nx
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


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
                                   basekV = self.dss_instance.Bus.kVBase(),
                                   latitude = self.dss_instance.Bus.Y(),
                                   longitude = self.dss_instance.Bus.X())
            
            bus_data.append(bus_coordinates)
        return bus_data   

class TransformerHandler:
    # to do: delta connected split phase primary is not handled yet
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
            splitphase_node_primary (dict[str,str]): A dictionary with primary node as key and associated phase as value
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
    

class NetworkHandler:
    # to do :-> extract feeder information to visualize it better
    def __init__(self, 
                 dss_instance: ModuleType, 
                 bus_names: Optional[list[str]] = None,
                 source: Optional[str] = None,
                 pdelement_handler: Optional[PDElementHandler] = None,
                 pdelements_data: Optional[list[dict[str,Union[int,str,float, npt.NDArray[np.complex128]]]]] = None) -> None:
        
        """Initialize a NetworkHandler instance. Create and modify the network as a graph (nodes and edges) for the distribution model 
        
        Args:
            dss_instance (ModuleType): redirected opendssdirect instance
            bus_names (Optional[list[str]]):Names of all the buses (nodes) in the distribution model. Defaults to None
            source (Optional[str], optional): Source node of the graph to build. Defaults to None.
            pdelement_handler (Optional[PDElementHandler], optional): Instance of PDElementHandler. Defaults to None.
            pdelements_data (Optional[list[dict[str,Union[int,str,float]]]], optional): All the required data of the pdelements(edges) 
            from PDElementHandler or provided by user in pdelements format. Defaults to None.
        """                      
        
        self.dss_instance = dss_instance 
        self.bus_names = bus_names
        self.source = source        
        self.pdelements_data = pdelements_data
        self.pdelement_handler = pdelement_handler
    
    def __network_input_validator(self) -> None:
        """Validates the data required to build a network

        Raises:
            NotImplementedError (elements): This is raised when pdelements data are not provided (either json or PDElementHandler)
            EOFError (source): This is raised when a source does not exist in the tree i.e. the bus of the distribution model
        """        
        if self.bus_names is None:
            logging.info("Bus names not provided. So extracting it from the base network.")
            self.bus_names = self.dss_instance.Circuit.AllBusNames()
        
        if self.pdelements_data is None and self.pdelement_handler is None:
            logging.warning("You need to provide either one of the following: pcelement list of dicts or PCElementHandler")
            raise NotImplementedError(
                "Please provide an element file (json) OR PDElement instance from PDElement handler to create the network."
                )
        
        if self.source is not None and self.source not in self.bus_names:
            logging.warning("The source must be one of the existing buses (nodes) in the distribution model.")
            raise EOFError(
                "Please provide a valid source. A source must be an existing bus (node) in the distribution model"
                )
    
    def __set_node_coordinates(self, network_tree: nx.DiGraph) -> None:
        """Sets the coordinates of each nodes as per the bus coordinates data

        Args:
            network_tree (nx.DiGraph): A directed graph (network tree in this case)
        """
        
        for node in network_tree.nodes():
            
            # need to set the nodes active before extracting their info 
            self.dss_instance.Circuit.SetActiveBus(node)
                        
            # be careful that X gives you lon and Y gives you lat
            network_tree.nodes[node]['lat'] = self.dss_instance.Bus.Y()
            network_tree.nodes[node]['lon'] = self.dss_instance.Bus.X()
    
        
    def build_network(self, all_pdelements: list[dict[str,Union[int,str,float, npt.NDArray[np.complex128]]]]) -> tuple[nx.Graph, list[str]]: 
        """Build the network from the pdelements data

        Args:
            all_pdelements (list[dict[str,Union[int,str,float, npt.NDArray[np.complex128]]]]): All the required data of the pdelements(edges) 
            from PDElementHandler or provided by user in pdelements format.

        Returns:
            network_graph (nx.Graph): Network graph (undirected) 
            normally_open_components (list[str]): Names of normally open pdelements (tie or virtual switches) 
        """         
        
        # initiate a network graph. Since the power flow occurs in any direction this should be undirected
        network_graph = nx.Graph()
        
        # extract normally open components
        normally_open_components = []
        
        # add lines as graph edges. initial version may have loops so we create graphs and then create trees
        for each_line in all_pdelements:
            if not each_line['is_open']:
                network_graph.add_edge(each_line['from_bus'],
                                       each_line['to_bus'], 
                                       # the remaining arguments are the data associated with each edges
                                       element=each_line['element'],
                                       is_switch=each_line['is_switch'])
            else:
                normally_open_components.append(each_line['name'])

        return network_graph, normally_open_components          
        
    def network_topology(self) -> tuple[nx.Graph, nx.DiGraph, list[str]]:
        """Create network topology including graph, trees, and open components

        Returns:
            network_graph (nx.Graph): Undirected network graph
            network_tree (list[str]): Directed network tree
            normally_open_components (list[str]): Names of normally open pdelements (tie or virtual switches)
        """
        
        # validate the data first i.e. whether pdelements were provided or not
        self.__network_input_validator()
        self.source = self.bus_names[0]
        
        # this can be user defined lines or can be extracted from the base network using PDElementHandler
        all_pdelements = self.pdelement_handler.get_pdelements() if not self.pdelements_data else self.pdelements_data
        
        network_graph, normally_open_components = self.build_network(all_pdelements)
        network_tree = nx.bfs_tree(network_graph, source=self.source)
        
        # add the bus corrdinates from dss to network tree
        self.__set_node_coordinates(network_tree)   
        
        return network_graph, network_tree, normally_open_components

class LoadHandler:
    # to do: condition for delta connected loads
    def __init__(self, 
                 dss_instance: ModuleType,
                 network_handler: Optional[NetworkHandler] = None,
                 transformer_handler: Optional[TransformerHandler] = None, 
                 include_secondary_network: Optional[bool] = False,                 
                 bus_names: Optional[list[str]] = None) -> None:
        
        """Initialize a LoadHandler instance. This instance deals with all the loads in the distribution system. 

        Args:
            dss_instance (ModuleType): redirected opendssdirect instance
            network_handler (Optional[NetworkHandler]): Directed network tree of the distribution model, Defaults to None
            transformer_handler (Optional[TransformerHandler]): Instance of TransformerHandler. Defaults to None.
            include_secondary_network (Optional[bool]): Whether the secondary network is to be considered or not, Defaults to False
            bus_names (Optional[list[str]]):Names of all the buses (nodes) in the distribution model
        """        
        
        self.dss_instance = dss_instance         
        self.network_handler = network_handler
        self.transformer_handler = transformer_handler
        self.include_secondary_network = include_secondary_network
        
        # since bus_names is required for any methods in LoadHandler, we rather check it in the initialization
        self.bus_names = self.dss_instance.Circuit.AllBusNames() if bus_names is None else bus_names        
        self.downstream_nodes_from_primary = None
        
        # validate if the required inputs are in existence
        self.__load_input_validator()

    def __load_input_validator(self) -> None:
            """This is to be checked in the future. Network and Transformer handler should be optional
            and only available if loads are to be referred to the primary."""
            
            if not self.include_secondary_network and (not self.transformer_handler and not self.network_handler):
                # if we do not want secondary and we do not pass any handlers then there must be an error
                logging.warning("You need to provide NetworkHandler() and TransformerHandler as arguments to LoadHandler()")
                raise NotImplementedError(
                "To refer the loads to primary, both NetworkHandler and TransformerHandler are required.")  
    
    def get_loads(self) -> None:
        
        if self.include_secondary_network:
            # get all loads as they appear in the secondary
            logging.info("Fetching the loads as they appear on the secondary")
            self.get_all_loads()
        else:
            # get primarry referred loads
            logging.info("Referring the loads back to the primary node of the distribution transformer.")
            self.get_primary_referred_loads()
             
        
        
    def get_all_loads(self) -> pd.DataFrame:
        """Extract load information for each bus(node) for each phase. This method extracts load on the exact bus(node) as 
        modeled in the distribution model, including secondary.

        Returns:
            load_per_phase(pd.DataFrame): Per phase load data in a pandas dataframe
        """
            
        num_buses = len(self.bus_names)

        # Initialize arrays for load_per_phase
        load_per_phase = {
            "name": [""] * num_buses,
            "bus": self.bus_names,
            "P1": np.zeros(num_buses),
            "Q1": np.zeros(num_buses),
            "P2": np.zeros(num_buses),
            "Q2": np.zeros(num_buses),
            "P3": np.zeros(num_buses),
            "Q3": np.zeros(num_buses)
        }

        loads_flag = self.dss_instance.Loads.First()

        while loads_flag:
            connected_buses = self.dss_instance.CktElement.BusNames()           
            
            # conductor power contains info on active and reactive power
            conductor_power = np.array(self.dss_instance.CktElement.Powers())
            nonzero_power_indices = np.where(conductor_power > 0)[0]
            nonzero_power = conductor_power[nonzero_power_indices]

            for buses in connected_buses:
                bus_split = buses.split(".")
                if (len(bus_split) == 4) or (len(bus_split) == 1):
                    
                    # three phase checker
                    connected_bus = bus_split[0]
                    bus_index = self.bus_names.index(connected_bus)  
                    load_per_phase["name"][bus_index] = self.dss_instance.Loads.Name()
                    P_values = nonzero_power[::2]   # Extract P values (every other element starting from the first)
                    Q_values = nonzero_power[1::2]  # Extract Q values (every other element starting from the second)
                    for phase_index in range(3):
                        load_per_phase[f"P{phase_index + 1}"][bus_index] += round(P_values[phase_index],2)
                        load_per_phase[f"Q{phase_index + 1}"][bus_index] += round(Q_values[phase_index],2)

                else:
                    # non three phase load
                    connected_bus, connected_phase_secondary = bus_split[0], bus_split[1:]
                    bus_index = self.bus_names.index(connected_bus)                    
                    load_per_phase["name"][bus_index] = self.dss_instance.Loads.Name()
                    P_values = nonzero_power[::2]  # Extract P values (every alternate element starting from the first)
                    Q_values = nonzero_power[1::2]  # Extract Q values (every alternate element starting from the second)
                    
                    for phase_index, phase in enumerate(connected_phase_secondary):
                        load_per_phase[f"P{phase}"][bus_index] += round(P_values[phase_index], 2)
                        load_per_phase[f"Q{phase}"][bus_index] += round(Q_values[phase_index], 2)

            loads_flag = self.dss_instance.Loads.Next()
        
        return pd.DataFrame(load_per_phase)

    
    def get_primary_referred_loads(self) -> pd.DataFrame:
        """Transfer all the secondary nodes to the primary corresponding to each split phase transformer. 
        Also returns the downstream nodes from the split phase transformers.

        Returns:
            primary_loads_df(pd.DataFrame): Per phase load data in a pandas dataframe with secondary transferred to primary
        """
        self.downstream_nodes_from_primary = []  
        _, network_tree, _ = self.network_handler.network_topology()
                
        # obtain the relation between the primary phase and secondary bus in splitphase transformer
        split_phase_primary = self.transformer_handler.get_splitphase_primary()
        
        # initially this is the secondary load but will be changed to reflect the primary load referral
        primary_loads_df = self.get_all_loads()
        
        for xfrmr_secondary_node, primary_phase in split_phase_primary.items():
            
            # here we get the predecessor of the secondary i.e. primary node
            xfrmr_primary_node = list(network_tree.predecessors(xfrmr_secondary_node))[0]
            
            # identify the secondary and primary bus indices so that loads are referred to primary 
            secondary_bus_index = self.bus_names.index(xfrmr_secondary_node)
            primary_bus_index = self.bus_names.index(xfrmr_primary_node)
            
            # however we still traverse downstream from the secondary as traversing from primary could follow other routes too
            xfrmr_downstream_nodes = nx.descendants(network_tree, xfrmr_secondary_node)
            
            # extend the xfmr secondary and downstream in the removal list for future
            # since we are aggregating these loads in the primary, removing them will reduce computational burden
            self.downstream_nodes_from_primary.extend(list(xfrmr_downstream_nodes)) 
            self.downstream_nodes_from_primary.extend([xfrmr_secondary_node])          
            
            for load_node in xfrmr_downstream_nodes:
                load_bus_index = self.bus_names.index(load_node)   
                # if np.any(loads_df.iloc[loads_df.index.get_loc(secondary_bus_index), 1:].to_numpy() > 0):            
                primary_loads_df.loc[primary_bus_index, f"P{primary_phase[0]}"] += (primary_loads_df["P1"][load_bus_index] +
                                                                                    primary_loads_df["P2"][load_bus_index])                
                # drop the secondaries from the dataframe
                primary_loads_df.drop(load_bus_index, inplace=True)  
                primary_loads_df.drop(secondary_bus_index, inplace=True)  
        
        # reset the loads dataframe to its original index
        primary_loads_df.reset_index(inplace=True, drop=True)
        
        return primary_loads_df    


class DSSManager:
    def __init__(self,
                 dssfile: str,
                 include_DERs: bool =True,
                 DER_pf: float = 0.9,
                 include_secondary_network: bool = False) -> None:
        
        logging.info(f'Initialize DSSManager')
        
        """Initialize a DSSManager instance. This instance manages all the components in the distribution system.

        Args:
            dssfile (str): path of the dss master file (currently only supports OpenDSS files)
            include_DERs (bool, optional): Check whether to include DERs or not. Defaults to True.
            DER_pf (float, optional): Constant power factor of DERs. Defaults to 0.9.
            include_secondary (bool, optional): Check whether to include secondary network or not. Defaults to False.
        """
        
        self.dssfile= dssfile
        self.dss = dss
        self.dss.Text.Command(f"Redirect {self.dssfile}")
        
        self.bus_names = self.dss.Circuit.AllBusNames()             # extract all bus names
        self.source = self.bus_names[0]                             # typically the first bus is the source
        self.include_DERs = include_DERs                            # variable to check whether to include DERs or not
        self.DERs = []                                              # variable to store information on DERs if included
        self.DER_pf = DER_pf                                        # constant power factor of DERs
        self.open_switches = []                                     # variable to store information on normally open switches
        self.include_secondary_network = include_secondary_network  # check whether to include secondary or not
        
        # initialize parsing process
        self.__initialize()
        
    
    def __initialize(self) -> None:
        """
        Initialize user-based preferences as well as DSS handlers (i.e. load, transformer, pdelements, and network)
        """        
        # if DERs are to be included then include virtual switches for DERs
        if self.include_DERs:
            self.DERs = self.__initializeDERs()
            logging.info("DERs virtual switches have been added successfully.")
        else:
            logging.info("DERs virtual switches are not included due to exclusion of DERs.")
        
        # initialize DSS handlers
        self.__initialize_dsshandlers()
        
                
    def __initializeDERs(self) -> None:   
        """
        Include or exclude virtual switches for DERs based on DER inclusion flag
        """         
        self.DERs = []
        generator_flag = self.dss.Generators.First()
        while generator_flag:                    
            self.dss.Text.Command('New Line.{virtual_DERswitch} phases=3 bus1={source_bus} bus2={gen_bus} switch=True r1=0.001 r0=0.001 x1=0.001 x0=0.001 C1=0 C0=0 length=0.001'.
                            format(virtual_DERswitch = self.dss.Generators.Name(),
                                    source_bus = self.source,
                                    gen_bus = self.dss.Generators.Bus1())
                            )
            self.DERs.append({
                'name': self.dss.Generators.Name(),
                'kW_rated': round(self.dss.Generators.kVARated() * self.DER_pf, 2),
                'bus': self.dss.Generators.Bus1(),
                'phases': self.dss.Generators.Phases()
                }) 
                
            generator_flag = self.dss.Generators.Next()
        
        # we also need to ensure that these switches are open as they are virtual switches
        for each_DERs in self.DERs:
            self.dss.Text.Command(f'Open Line.{each_DERs["name"]}')    
    
    def __initialize_dsshandlers(self) -> None: 
        """Initialize all the DSS Handlers
        """               
        
        # bus_handler is currently not being used here but kept here for future usage    
        self.bus_handler = BusHandler(self.dss)
        self.transformer_handler = TransformerHandler(self.dss)
        self.pdelement_handler = PDElementHandler(self.dss)
        self.network_handler = NetworkHandler(self.dss,
                                              pdelement_handler=self.pdelement_handler)
        
        if self.include_secondary_network:
            logging.info("Considering entire system including secondary networks")
            self.load_handler = LoadHandler(self.dss)
        else:
            # if primary loads are to be referred then we must pass network and transformer handlers 
            logging.info("Considering primary networks and aggregating loads by referring them to the primary node")
            self.load_handler = LoadHandler(self.dss,
                                            include_secondary_network = self.include_secondary_network,
                                            network_handler = self.network_handler,
                                            transformer_handler = self.transformer_handler
                                            )
        logging.info(f'Successfully instantiated required handlers from {self.dssfile}.')
        
    def parse_dss(self) -> None:
        self.bus_data = self.bus_handler.get_buses() 
        self.transformer_data = self.transformer_handler.get_transformers() 
        self.pdelements_data = self.pdelement_handler.get_pdelements() 
        self.network_graph, self.network_tree, self.normally_open_components = self.network_handler.network_topology()
        self.load_data = self.load_handler.get_loads()        
        breakpoint()

if __name__ == "__main__":
    dss_data = DSSManager(r"../../../distributiondata/ieee9500_dss/Master-unbal-initial-config.dss")
    
    breakpoint()
