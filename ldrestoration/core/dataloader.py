from __future__ import annotations
from typing import Any, TYPE_CHECKING
import json
from networkx.readwrite import json_graph
from pathlib import Path
import logging
import pandas as pd

if TYPE_CHECKING:
    import networkx as nx


from ldrestoration.utils.loggerconfig import setup_logging
setup_logging()
logger = logging.getLogger(__name__)
    

class DataLoader:
    def __init__(self, data_folder_path: str) -> None:
        """Class to loads the required data for the optimization model. 
        Note: the files must be created via ldrestoration.dssparser.dssparser.DSSManager.saveparseddss module
        as this function expects the files to have the same name and type as saved by this module. 

        Args:
            data_folder_path (str): Path or folder containing the required files
        """ 
        try:       
            self.data_folder_path = data_folder_path
        except FileNotFoundError:
            logger.error(f'{self.data_folder_path} does not exist')
            raise FileNotFoundError(f"{self.data_folder_path} does not exist. Please ensure that the path to the folder is correct.") 
    
    def load_circuit_data(self) -> dict[str, Any]:
        """Loads the essential circuit data. 
        
        Returns:
            dict[str, Any]: key is the metadata and value corresponds to the data for specific metadata
            eg. {"substation": "sourcebus"}
        """   
        
        try:
            with open(Path(self.data_folder_path) / 'circuit_data.json', 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {Path(self.data_folder_path) / 'circuit_data.json'} does not exist. "
                                    " Please make sure the folder contains files as created by the dataparser module.")

    
    def load_network_graph(self) -> nx.Graph:
        """Loads the networkx graph. 
        
        Returns:
            nx.Graph: networkx graph of the model 
        """   
        
        try:
            with open(Path(self.data_folder_path) / 'network_graph_data.json', 'r') as file:
                network_graph_file = json.load(file)
                network_graph = json_graph.node_link_graph(network_graph_file)
            return network_graph
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {Path(self.data_folder_path) / 'network_graph_data.json'} does not exist. "
                                    " Please make sure the folder contains files as created by the dataparser module.")
    
    def load_network_tree(self) -> nx.DiGraph:
        """Loads the networkx graph. 
        
        Returns:
            nx.DiGraph: networkx tree of the model 
        """   
        
        try:
            with open(Path(self.data_folder_path) / 'network_tree_data.json', 'r') as file:
                network_tree_file = json.load(file)
                network_tree = json_graph.node_link_graph(network_tree_file)
            return network_tree
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {Path(self.data_folder_path) / 'network_tree_data.json'} does not exist. "
                                    " Please make sure the folder contains files as created by the dataparser module.")
        
    def load_network_loads(self) -> pd.DataFrame:
        """Loads the network load i.e. active and reactive power demand. 
        
        Returns:
            pd.DataFrame: DataFrame of overall load (active and reactive demand) of each node 
        """   
        
        try:
            loads = pd.read_csv(Path(self.data_folder_path) / 'load_data.csv')
            return loads
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {Path(self.data_folder_path) / 'load_data.csv'} does not exist."
                                    " Please make sure the folder contains files as created by the dataparser module.")
    
    def load_network_ders(self) -> pd.DataFrame | None:
        """Loads the DERs (distributed energy resources) if available. 
        
        Returns:
            pd.DataFrame: DataFrame of DERs available in the network.
        """   
        
        try:
            DERs = pd.read_csv(Path(self.data_folder_path) / 'DERs.csv', delimiter=',')
            return DERs
        except pd.errors.EmptyDataError:
            logger.warning("Empty file detected. This means that no DERs are detected in the base system.")
            return None
        except FileNotFoundError:
            logger.warning("The DER file is missing. Please either pass include_DERs=True in DSSManager module or provide DER details."
                           "The latter will be included in the future ...")
            return None
        
    def load_network_normally_open_components(self) -> pd.DataFrame:
        """Loads the normally open components (tie switches and virtual DER switches) in the network, if available. 
        
        Returns:
            pd.DataFrame: DataFrame of normally open components in the network.
        """   
        
        try:
            normally_open_components = pd.read_csv(Path(self.data_folder_path) / 'normally_open_components.csv')
            return normally_open_components
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {Path(self.data_folder_path) / 'normally_open_components.csv'} does not exist."
                                    " Please make sure the folder contains files as created by the dataparser module.")
    
    def load_network_pdelements(self) -> pd.DataFrame:
        """Loads the power delivery elements (line, transformer, switches, reactors) in the network. 
        
        Returns:
            pd.DataFrame: DataFrame of pdelements in the network.
        """   
        
        try:
            pdelements = pd.read_csv(Path(self.data_folder_path) / 'pdelements_data.csv')
            return pdelements
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {Path(self.data_folder_path) / 'pdelements_data.csv'} does not exist."
                                    " Please make sure the folder contains files as created by the dataparser module.")
    
    def load_data(self) -> dict[str, Any]:
        """Loads the required data for the optimization model. 
        Note: the files must be created via ldrestoration.dssparser.dssparser.DSSManager.saveparseddss module
        as this function expects the files to have the same name and type as saved by this module. 

        Returns:
            [dict]: returns the dictionary of overall data 
        """    
        return dict(
            network_graph=self.load_network_graph(),
            network_tree=self.load_network_tree(),
            loads=self.load_network_loads(),                        
            DERs=self.load_network_ders(),
            normally_open_components=self.load_network_normally_open_components(),
            pdelements=self.load_network_pdelements(),
            circuit_data = self.load_circuit_data()                       
            )