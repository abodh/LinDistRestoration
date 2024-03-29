from __future__ import annotations

from types import ModuleType
from typing import Optional
import numpy as np
import pandas as pd
import networkx as nx
import logging
from functools import cache

from ldrestoration.dssparser.networkhandler import NetworkHandler
from ldrestoration.dssparser.transformerhandler import TransformerHandler

from ldrestoration.utils.decors import timethis
from ldrestoration.utils.loggerconfig import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class LoadHandler:
    """LoadHandler deals with all the loads in the distribution system. 
    When `include_secondary_network=False`, all the secondary loads are referred back to their primary.  

    Args:
        dss_instance (ModuleType): redirected opendssdirect instance
        network_handler (Optional[NetworkHandler]): Directed network tree of the distribution model, Defaults to None
        transformer_handler (Optional[TransformerHandler]): Instance of TransformerHandler. Defaults to None.
        include_secondary_network (Optional[bool]): Whether the secondary network is to be considered or not, Defaults to False
        bus_names (Optional[list[str]]):Names of all the buses (nodes) in the distribution model
    
    Note:
        * In OpenDSS, the phases information of loads are lost on the secondary side of the split-phase transformers. Hence, each of the loads are traced back to their 
        nearest transformer to identify the corresponding phase. For delta primary, the loads are equally distributed to each phase. 
    
    To do:
        * The current version does not address phase wise load decoupling for delta connected loads. It will be incorporated in the future releases.
    
    """     

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

    @timethis
    def __load_input_validator(self) -> None:
            """This is to be checked in the future. Network and Transformer handler should be optional
            and only available if loads are to be referred to the primary."""
            
            if not self.include_secondary_network and (not self.transformer_handler and not self.network_handler):
                # if we do not want secondary and we do not pass any handlers then there must be an error
                logger.warning("You need to provide NetworkHandler() and TransformerHandler as arguments to LoadHandler()")
                raise NotImplementedError(
                "To refer the loads to primary, both NetworkHandler and TransformerHandler are required."
                )  
    
    @timethis
    def get_loads(self) -> pd.DataFrame:
           
        if self.include_secondary_network:
            # get all loads as they appear in the secondary
            logger.info("Fetching the loads as they appear on the secondary")
            return self.get_all_loads()
        else:
            # get primarry referred loads
            logger.info("Referring the loads back to the primary node of the distribution transformer.")
            return self.get_primary_referred_loads()             
    
    @property
    @cache
    def bus_names_to_index_map(self) -> dict[str,int]:
        """each of the bus mapped to its corresponding index in the bus names list

        Returns:
            dict[str,int]: dictionary with key as bus names and value as its index
        """        
        return {bus:index for index,bus in enumerate(self.bus_names)}
        
    @timethis    
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
            nonzero_power_indices = np.where(conductor_power != 0)[0]
            nonzero_power = conductor_power[nonzero_power_indices]

            for buses in connected_buses:
                bus_split = buses.split(".")
                if (len(bus_split) == 4) or (len(bus_split) == 1):
                    
                    # three phase checker
                    connected_bus = bus_split[0]
                    bus_index = self.bus_names_to_index_map[connected_bus]
                    load_per_phase["name"][bus_index] = self.dss_instance.Loads.Name()
                    P_values = nonzero_power[::2]   # Extract P values (every other element starting from the first)
                    Q_values = nonzero_power[1::2]  # Extract Q values (every other element starting from the second)
                    for phase_index in range(3):
                        load_per_phase[f"P{phase_index + 1}"][bus_index] += round(P_values[phase_index],2)
                        load_per_phase[f"Q{phase_index + 1}"][bus_index] += round(Q_values[phase_index],2)

                else:
                    # non three phase load
                    connected_bus, connected_phase_secondary = bus_split[0], bus_split[1:]
                    bus_index = self.bus_names_to_index_map[connected_bus]                  
                    load_per_phase["name"][bus_index] = self.dss_instance.Loads.Name()
                    P_values = nonzero_power[::2]  # Extract P values (every alternate element starting from the first)
                    Q_values = nonzero_power[1::2]  # Extract Q values (every alternate element starting from the second)
                    
                    for phase_index, phase in enumerate(connected_phase_secondary):
                        load_per_phase[f"P{phase}"][bus_index] += round(P_values[phase_index], 2)
                        load_per_phase[f"Q{phase}"][bus_index] += round(Q_values[phase_index], 2)
                        
            loads_flag = self.dss_instance.Loads.Next()
        
        return pd.DataFrame(load_per_phase)        
    
    @timethis
    def get_primary_referred_loads(self) -> pd.DataFrame:
        """Transfer all the secondary nodes to the primary corresponding to each split phase transformer. 
        Also returns the downstream nodes from the split phase transformers.

        Returns:
            primary_loads_df(pd.DataFrame): Per phase load data in a pandas dataframe with secondary transferred to primary
        """
        # keep track of all downstream nodes from primary. This is for removal from network as well,
        # since we are aggregating these loads in the primary, removing them will reduce computational burden
        self.downstream_nodes_from_primary = set()  
        
        # get access to the network tree from the topology
        _, network_tree, _ = self.network_handler.network_topology()
                
        # obtain the relation between the primary phase and secondary bus in splitphase transformer
        # this obtains a dictionary with secondary nodes s key and their associated phase info as value
        split_phase_primary = self.transformer_handler.get_splitphase_primary()
        
        # initially this is the secondary load but will be changed to reflect the primary load referral
        primary_loads_df = self.get_all_loads()   
        
        for xfrmr_secondary_node, primary_phase in split_phase_primary.items():
            
            # here we get the predecessor of the secondary i.e. primary node
            xfrmr_primary_node = list(network_tree.predecessors(xfrmr_secondary_node))[0]
            
            # identify the secondary and primary bus indices so that loads are referred to primary             
            secondary_bus_index = self.bus_names_to_index_map[xfrmr_secondary_node]
            primary_bus_index = self.bus_names_to_index_map[xfrmr_primary_node]
            
            # however we still traverse downstream from the secondary as traversing from primary could follow other routes too
            xfrmr_downstream_nodes = nx.descendants(network_tree, xfrmr_secondary_node)
            xfrmr_downstream_nodes.add(xfrmr_secondary_node)
            
            # add all the nodes downstream from the transformer's primary (including secondary)
            self.downstream_nodes_from_primary.update(xfrmr_downstream_nodes)      
            
            # now we refer all the downstream node loads (if available) to the primary and remove them from the load df
            # this reduces computational burden when not dealing with the secondaries
            for xfrmr_downstream_node in xfrmr_downstream_nodes:   
                try:
                    downstream_node_index = self.bus_names_to_index_map[xfrmr_downstream_node]
                except KeyError:
                    logger.error(f"Invalid load name {xfrmr_downstream_node}")
                    raise KeyError(f"{xfrmr_downstream_node} is not a valid node name.")
                        
                primary_loads_df.loc[primary_bus_index, f"P{primary_phase[0]}"] += (primary_loads_df["P1"][downstream_node_index] +
                                                                                    primary_loads_df["P2"][downstream_node_index]) 
                primary_loads_df.loc[primary_bus_index, f"Q{primary_phase[0]}"] += (primary_loads_df["Q1"][downstream_node_index] +
                                                                                    primary_loads_df["Q2"][downstream_node_index])                
                
                primary_loads_df.loc[primary_bus_index, f"name"] = primary_loads_df["name"][downstream_node_index]

                # drop the secondaries from the dataframe
                primary_loads_df.drop(downstream_node_index, inplace=True) 
                # primary_loads_df.drop(secondary_bus_index, inplace=True)  
                    
        # reset the loads dataframe to its original index
        primary_loads_df.reset_index(inplace=True, drop=True)

        return primary_loads_df 