from typing import Any
import pandas as pd
import json
from networkx.readwrite import json_graph
from pathlib import Path
import logging

from ldrestoration.utils.loggerconfig import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

def load_data(data_folder_path: str) -> dict[str,Any]:
    """Loads the required data for the optimization model. 
    Note: the files must be created via ldrestoration.dssparser.dssparser.DSSManager.saveparseddss module
    as this function expects the files to have the same name and type as saved by this module. 

    Args:
        data_folder_path (str): Path or folder containing the required files

    Returns:
        dict[str,Any]: dict with specified name as keys and the data as value 
    """    
    data_dict = {}

    try:
        with open(Path(data_folder_path) / 'network_graph_data.json', 'r') as file:
            network_graph_file = json.load(file)
            network_graph = json_graph.node_link_graph(network_graph_file)
        data_dict['network_graph'] = network_graph
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {Path(data_folder_path) / 'network_graph_data.json'} does not exist. "
                                " Please make sure the folder contains files as created by the dataparser module.")

    try:
        with open(Path(data_folder_path) / 'network_tree_data.json', 'r') as file:
            network_tree_file = json.load(file)
            network_tree = json_graph.node_link_graph(network_tree_file)
        data_dict['network_tree'] = network_tree
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {Path(data_folder_path) / 'network_tree_data.json'} does not exist. "
                                " Please make sure the folder contains files as created by the dataparser module.")

    try:
        loads = pd.read_csv(Path(data_folder_path) / 'load_data.csv')
        data_dict['loads'] = loads
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {Path(data_folder_path) / 'load_data.csv'} does not exist."
                                " Please make sure the folder contains files as created by the dataparser module.")

    try:
        DERs = pd.read_csv(Path(data_folder_path) / 'DERs.csv')
        data_dict['DERs'] = DERs
    except FileNotFoundError:
        logger.warning("This is to notify that DERs are excluded and if you want them to be included "
                       "please either pass include_DERs=True in DSSManager module or provide DER details."
                       "The latter will be included in the future ...")
        
    try:
        normally_open_components = pd.read_csv(Path(data_folder_path) / 'normally_open_components.csv')
        data_dict['normally_open_components'] = normally_open_components
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {Path(data_folder_path) / 'normally_open_components.csv'} does not exist."
                                " Please make sure the folder contains files as created by the dataparser module.")

    try:
        pdelements = pd.read_csv(Path(data_folder_path) / 'pdelements_data.csv')
        data_dict['pdelements'] = pdelements
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {Path(data_folder_path) / 'pdelements_data.csv'} does not exist."
                                " Please make sure the folder contains files as created by the dataparser module.")

    return data_dict

if __name__=='__main__':
    data_dict=load_data(r"../../examples/parsed_data/dssdatatocsv_01-17-2024_12-38-43/")
    breakpoint()