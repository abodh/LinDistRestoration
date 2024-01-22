from __future__ import annotations
import opendssdirect as dss 
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from networkx.readwrite import json_graph
import json
import logging

from ldrestoration.utils.decors import timethis
from ldrestoration.utils.loggerconfig import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

from ldrestoration.dssparser.bushandler import BusHandler
from ldrestoration.dssparser.transformerhandler import TransformerHandler
from ldrestoration.dssparser.pdelementshandler import PDElementHandler
from ldrestoration.dssparser.networkhandler import NetworkHandler
from ldrestoration.dssparser.loadhandler import LoadHandler


class DSSManager:
    def __init__(self,
                 dssfile: str,
                 include_DERs: bool =True,
                 DER_pf: float = 0.9,
                 include_secondary_network: bool = False) -> None:
        
        logging.info(f'Initializing DSSManager')
        
        """Initialize a DSSManager instance. This instance manages all the components in the distribution system.

        Args:
            dssfile (str): path of the dss master file (currently only supports OpenDSS files)
            include_DERs (bool, optional): Check whether to include DERs or not. Defaults to True.
            DER_pf (float, optional): Constant power factor of DERs. Defaults to 0.9.
            include_secondary (bool, optional): Check whether to include secondary network or not. Defaults to False.
        """
        
        self.dss = dss
        self.dssfile = dssfile
        # opendss direct checks for filenotfound exception so we do not require any exception here
        self.dss.Text.Command(f"Redirect {self.dssfile}")
        
        # initialize other attributes
        self.include_DERs = include_DERs                            # variable to check whether to include DERs or not
        self.DER_pf = DER_pf                                        # constant power factor of DERs
        self.include_secondary_network = include_secondary_network  # check whether to include secondary or not
        self.DERs = None                                            # variable to store information on DERs if included
                
        # initialize parsing process variables and handlers
        self.__initialize()
    
    @property
    def bus_names(self) -> list[str]:
        """Access all the bus (node) names from the circuit 

        Returns:
            list[str]: list of all the bus names
        """        
        return self.dss.Circuit.AllBusNames()
    
    
    @property
    def basekV_LL(self) -> float:
        """Returns basekV (line to line) of the circuit based on the sourcebus

        Returns:
            float: base kV of the circuit as referred to the source bus
        """         
        # make the source bus active before accessing the base kV since there is no provision to get base kV of circuit
        self.dss.Circuit.SetActiveBus(self.source)
        return round(self.dss.Bus.kVBase() * np.sqrt(3), 2)
    
    @property
    def source(self) -> str:
        """source bus of the circuit. 

        Returns:
            str: returns the source bus of the circuit
        """     
        # typically the first bus is the source bus   
        return self.bus_names[0]
        
    @timethis    
    def __initialize(self) -> None:
        """
        Initialize user-based preferences as well as DSS handlers (i.e. load, transformer, pdelements, and network)
        """        
        # if DERs are to be included then include virtual switches for DERs
        if self.include_DERs:
            self.__initializeDERs()
            logging.info(f"DERs virtual switches have been added successfully. The current version assumes a constant power factor of DERs; DERs power factor = {self.DER_pf}")
        else:
            logging.info("DERs virtual switches are not included due to exclusion of DERs.")
        
        # initialize DSS handlers
        self.__initialize_dsshandlers()
        
        # initialize data variables (these will be dynamically updated through handlers)
        self.bus_data = None
        self.transformer_data = None
        self.pdelements_data = None 
        self.network_graph = None
        self.network_tree = None
        self.normally_open_components = None
        self.load_data = None 
        
    @timethis            
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
                'connected_bus': self.dss.Generators.Bus1(),
                'phases': self.dss.Generators.Phases()
                }) 
                
            generator_flag = self.dss.Generators.Next()
        
        # we also need to ensure that these switches are open as they are virtual switches
        for each_DERs in self.DERs:
            self.dss.Text.Command(f'Open Line.{each_DERs["name"]}')   
        
        self.dss.Solution.Solve()
    
    @timethis
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
            self.load_handler = LoadHandler(self.dss,
                                            include_secondary_network = self.include_secondary_network)
        else:
            # if primary loads are to be referred then we must pass network and transformer handlers 
            logging.info("Considering primary networks and aggregating loads by referring them to the primary node")
            self.load_handler = LoadHandler(self.dss,
                                            include_secondary_network = self.include_secondary_network,
                                            network_handler = self.network_handler,
                                            transformer_handler = self.transformer_handler
                                            )
        logging.info(f'Successfully instantiated required handlers from "{self.dssfile}"')
    
    @timethis    
    def parsedss(self) -> None:
        """Parse required data from the handlers to respective class variables
        """        
        self.bus_data = self.bus_handler.get_buses() 
        self.transformer_data = self.transformer_handler.get_transformers() 
        self.pdelements_data = self.pdelement_handler.get_pdelements() 
        self.network_graph, self.network_tree, self.normally_open_components = self.network_handler.network_topology()
        self.load_data = self.load_handler.get_loads() 
        
        if not self.include_secondary_network:
            logging.info(f'Excluding secondaries from final tree, graph configurations, and pdelements.')
            self.network_tree.remove_nodes_from(self.load_handler.downstream_nodes_from_primary)
            self.network_graph.remove_nodes_from(self.load_handler.downstream_nodes_from_primary)
            self.pdelements_data = [items for items in self.pdelements_data
                                    if items['from_bus'] not in self.load_handler.downstream_nodes_from_primary and
                                    items['to_bus'] not in self.load_handler.downstream_nodes_from_primary]           
        logging.info(f'Successfully parsed the required data from all handlers.')
    
    @timethis    
    def saveparseddss(self,
                      folder_name: str = f"parsed_data",
                      folder_exist_ok: bool = False) -> None:
        """Saves the parsed data from all the handlers 

        Args:
            folder_name (str, optional): Name of the folder to save the data in. Defaults to "dssdatatocsv"_<current system date>.
            folder_exist_ok (bool, optional): Boolean to check if folder rewrite is ok. Defaults to False.
        """        
        
        # check if parsedss is run before saving these files
        if self.bus_data is None:
            logger.error("Please run DSSManager.parsedss() to parse the data and then run this function to save the files.")
            raise NotImplementedError(f"Data variables are empty. You must run {__name__}.DSSManager.parsedss() to extract the data before saving them.")
        
        # check if the path already exists. This prevent overwrite
        try:
            Path(folder_name).mkdir(parents=True, exist_ok=folder_exist_ok)  
        except FileExistsError:
            logger.error("The folder already exists and the module is attempting to rewrite the data in the folder. Either provide a path in <folder_name> or mention <folder_exist_ok=True> to rewrite the existing files.")
            raise FileExistsError("The folder or files already exist. Please provide a non-existent path.")

        # save all the data in the new folder 
        # the non-networkx data are all saved as dataframe in csv
        pd.DataFrame(self.bus_data).to_csv(f'{folder_name}/bus_data.csv', index=False)
        pd.DataFrame(self.transformer_data).to_csv(f'{folder_name}/transformer_data.csv', index=False)
        pd.DataFrame(self.pdelements_data).to_csv(f'{folder_name}/pdelements_data.csv', index=False)
        pd.DataFrame(self.load_data).to_csv(f'{folder_name}/load_data.csv', index=False)
        pd.DataFrame(self.normally_open_components, columns=['normally_open_components']).to_csv(f'{folder_name}/normally_open_components.csv', index=False)
        
        if self.DERs is not None:
            pd.DataFrame(self.DERs).to_csv(f'{folder_name}/DERs.csv', index=False)
        
        # the networkx data is saved as a serialized JSON
        network_graph_data = json_graph.node_link_data(self.network_graph)
        network_tree_data = json_graph.node_link_data(self.network_tree)
        
        with open(f'{folder_name}/network_graph_data.json', 'w') as file:
            json.dump(network_graph_data, file)

        with open(f'{folder_name}/network_tree_data.json', 'w') as file:
            json.dump(network_tree_data, file)
            
        logger.info('Successfully saved the required files.')

@timethis
def main() -> None:
    dss_data = DSSManager(r"../../examples/test_cases/ieee9500_dss/Master-unbal-initial-config.dss",
                          include_DERs=True,
                          include_secondary_network=False)
    dss_data.parsedss()
    dss_data.saveparseddss() 
            
if __name__ == '__main__':
    main()    
    