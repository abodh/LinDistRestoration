from types import ModuleType
from typing import Union, Optional
import numpy.typing as npt
import numpy as np
import networkx as nx
import logging

from ldrestoration.utils.decors import timethis
from ldrestoration.dssparser.pdelementshandler import PDElementHandler
from ldrestoration.utils.logger_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

class NetworkHandler:
    # to do :-> extract feeder information to visualize it better
    # to do : add a method to remove list of edges provided as an argument
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
            logger.info("Bus names not provided. So extracting it from the base network.")
            self.bus_names = self.dss_instance.Circuit.AllBusNames()
        
        if self.pdelements_data is None and self.pdelement_handler is None:
            logger.warning("You need to provide either one of the following: pcelement list of dicts or PCElementHandler")
            raise NotImplementedError(
                "Please provide an element file (json) OR PDElement instance from PDElement handler to create the network."
                )
        
        if self.source is not None and self.source not in self.bus_names:
            logger.warning("The source must be one of the existing buses (nodes) in the distribution model.")
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