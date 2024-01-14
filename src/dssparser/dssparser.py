import opendssdirect as dss
import pandas as pd
import numpy as np
import networkx as nx
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

from decors import timethis

class DSSpy:        
    def __init__(self,
                 dssfile,
                 DER_inclusions=True,
                 secondary_sections=False):

        dss.Text.Command(f"Redirect {dssfile}")
        
        # instantiate dss data     
        self.loads = dss.Loads
        self.transformers = dss.Transformers
        self.buses = dss.Circuit.AllBusNames()
        self.source = self.buses[0]
        self.basekV_LL = dss.Vsources.BasekV() * np.sqrt(3)
        
        # whether we are considering secondary sides of the distribution transformer or not
        self.secondary_sections = secondary_sections
        
        # if DERs are included or not
        if DER_inclusions:
            self.DERs = self.__DER_inclusion()     
            logging.info("The DERs virtual switches have been added successfully.")

        # load all the lines and elements and this comes after the DER switches are introduced
        self.lines = dss.Lines
        self.linecodes = dss.LineCodes
        self.pdelements = dss.PDElements                   
     
    def __DER_inclusion(self):    
        self.DERs = []
        generator_flag = dss.Generators.First()
        while generator_flag:                    
            dss.Text.Command('New Line.{virtual_DERswitch} phases=3 bus1={source_bus} bus2={gen_bus} switch=True r1=0.001 r0=0.001 x1=0.001 x0=0.001 C1=0 C0=0 length=0.001'.
                            format(virtual_DERswitch = dss.Generators.Name(),
                                    source_bus = self.source,
                                    gen_bus = dss.Generators.Bus1())
                            )
            self.DERs.append({
                'name': dss.Generators.Name(),
                'kW_rated': round(dss.Generators.kVARated() * 0.9, 2),
                'bus': dss.Generators.Bus1(),
                'phases': dss.Generators.Phases()
                }) 
                
            generator_flag = dss.Generators.Next()
        
        # we also need to ensure that these switches are open as they are virtual switches
        for each_DERs in self.DERs:
            dss.Text.Command(f'Open Line.{each_DERs["name"]}')     
            
        return self.DERs
    
    def __get_zmatrix(self):
        
        if ((len(dss.CktElement.BusNames()[0].split('.')) == 4) or 
            (len(dss.CktElement.BusNames()[0].split('.')) == 1)):
            
            # this is the condition check for three phase since three phase is either represented by bus_name.1.2.3 or bus_name    
            z_matrix = np.array(self.lines.RMatrix()) + 1j * np.array(self.lines.XMatrix())
            z_matrix = z_matrix.reshape(3, 3)
            
            return z_matrix
        
        else:
            
            # for other than 3 phases            
            active_phases = [int(phase) for phase in dss.CktElement.BusNames()[0].split('.')[1:]]
            z_matrix = np.zeros((3, 3), dtype=complex)
            r_matrix = self.lines.RMatrix()
            x_matrix = self.lines.XMatrix()
            counter = 0
            for _, row in enumerate(active_phases):
                for _, col in enumerate(active_phases):
                    z_matrix[row - 1, col - 1] = complex(r_matrix[counter], x_matrix[counter])
                    counter = counter + 1

            return z_matrix
                
    def get_splitphase_primary(self):    
        '''
        get the primary phase info from split phase transformers to refer all loads to the primary
        
        '''    
        splitphase_node_primary = {}                
        transformer_flag = self.transformers.First()        
        while transformer_flag:
            
            if (dss.CktElement.NumPhases() != 3) and self.transformers.NumWindings() == 3:
                bus_name = dss.CktElement.BusNames()[1].split('.')[0]
                bus_phases = dss.CktElement.BusNames()[0].split('.')[1:]
                
                if bus_name not in splitphase_node_primary:
                    splitphase_node_primary[bus_name] = bus_phases

            transformer_flag = dss.Transformers.Next()
            
        return splitphase_node_primary
    
    def network_topology(self,
                         source=None,
                         element_data=None):
        
        # to do :-> extract feeder information to visualize it better
        
        # check if there is a user defined source and if not use default dss source 
        if source is None:
            source = self.source        
        
        # this can be user defined lines or can be extracted from the base network
        all_lines = self.get_pdelements() if not element_data else element_data
           
        # initiate a network graph. Since the power flow occurs in any direction this should be undirected
        network_graph = nx.Graph()
        
        # add lines as graph edges. initial version may have loops so we create graphs and then create trees
        for each_line in all_lines:
            if not each_line['is_open']:
                network_graph.add_edge(each_line['from_bus'], 
                                       each_line['to_bus'], 
                                       # the remaining arguments are the data associated with each edges
                                       element=each_line['element'],
                                       is_switch=each_line['is_switch'])
            else:
                normally_open_components = each_line['name']

        network_tree = nx.bfs_tree(network_graph, source=source)
        
        for node in network_tree.nodes():
            # need to set the nodes active before extracting their info 
            dss.Circuit.SetActiveBus(node)
                        
            # be careful that X gives you lon and Y gives you lat
            network_tree.nodes[node]['lat'] = dss.Bus.Y()
            network_tree.nodes[node]['lon'] = dss.Bus.X()

        return network_tree, normally_open_components    
    
    def get_transferred_loads(self):
        
        all_buses = self.buses  
        downstream_nodes_removal = []  
        
        # obtain the network tree
        network_tree, _ = self.network_topology()
                
        # obtain the relation between the primary phase and secondary bus in splitphase transformer
        split_phase_primary = self.get_splitphase_primary()
        loads_df = self.get_loads()
        
        for xfrmr_secondary_node, primary_phase in split_phase_primary.items():
            
            # here we get the predecessor of the secondary i.e. primary node
            xfrmr_primary_node = list(network_tree.predecessors(xfrmr_secondary_node))[0]
            
            # identify the secondary and primary bus indices so that loads are referred to primary 
            secondary_bus_index = all_buses.index(xfrmr_secondary_node)
            primary_bus_index = all_buses.index(xfrmr_primary_node)
            
            # however we still traverse downstream from the secondary as traversing from primary could follow other routes too
            xfrmr_downstream_nodes = nx.descendants(network_tree, xfrmr_secondary_node)
            
            # extend the xfmr secondary and downstream in the removal list for future
            # since we are aggregating these loads in the primary, removing them will reduce computational burden
            downstream_nodes_removal.extend(list(xfrmr_downstream_nodes)) 
            downstream_nodes_removal.extend([xfrmr_secondary_node])          
            
            for load_node in xfrmr_downstream_nodes:
                load_bus_index = all_buses.index(load_node)   
                # if np.any(loads_df.iloc[loads_df.index.get_loc(secondary_bus_index), 1:].to_numpy() > 0):            
                loads_df.loc[primary_bus_index, f"P{primary_phase[0]}"] += (loads_df["P1"][load_bus_index] +
                                                                            loads_df["P2"][load_bus_index])                
                # drop the secondaries from the dataframe
                loads_df.drop(load_bus_index, inplace=True)  
                loads_df.drop(secondary_bus_index, inplace=True)  
        
        # reset the loads dataframe to its original index
        loads_df.reset_index(inplace=True, drop=True)
        
        # update the tree to remove the secondaries from the tree
        network_tree.remove_nodes_from(downstream_nodes_removal)
        
        return loads_df, downstream_nodes_removal
        
    def get_loads(self):
        
        num_buses = len(self.buses)

        # Initialize arrays for load_per_phase
        load_per_phase = {
            "name": [""] * num_buses,
            "bus": self.buses,
            "P1": np.zeros(num_buses),
            "Q1": np.zeros(num_buses),
            "P2": np.zeros(num_buses),
            "Q2": np.zeros(num_buses),
            "P3": np.zeros(num_buses),
            "Q3": np.zeros(num_buses)
        }

        loads_flag = self.loads.First()

        while loads_flag:
            connected_buses = dss.CktElement.BusNames()           
            
            # conductor power contains info on active and reactive power
            conductor_power = np.array(dss.CktElement.Powers())
            nonzero_power_indices = np.where(conductor_power > 0)[0]
            nonzero_power = conductor_power[nonzero_power_indices]

            for buses in connected_buses:
                bus_split = buses.split(".")
                if (len(bus_split) == 4) or (len(bus_split) == 1):
                    
                    # three phase checker
                    connected_bus = bus_split[0]
                    bus_index = self.buses.index(connected_bus)  
                    load_per_phase["name"][bus_index] = dss.Loads.Name()
                    P_values = nonzero_power[::2]  # Extract P values (every other element starting from the first)
                    Q_values = nonzero_power[1::2]  # Extract Q values (every other element starting from the second)
                    for phase_index in range(3):
                        load_per_phase[f"P{phase_index + 1}"][bus_index] += round(P_values[phase_index],2)
                        load_per_phase[f"Q{phase_index + 1}"][bus_index] += round(Q_values[phase_index],2)

                else:
                    # non three phase load
                    connected_bus, connected_phase_secondary = bus_split[0], bus_split[1:]
                    
                    # if len(actual_phases) == 1:
                    #     connected_phase = actual_phases 
                    # else:
                    #     connected_phase = actual_phases
                    #     assert False, "currently not supported for delta connected load. please stay tuned for update\n"

                    bus_index = self.buses.index(connected_bus)                    
                    load_per_phase["name"][bus_index] = dss.Loads.Name()
                    P_values = nonzero_power[::2]  # Extract P values (every other element starting from the first)
                    Q_values = nonzero_power[1::2]  # Extract Q values (every other element starting from the second)
                    for phase_index, phase in enumerate(connected_phase_secondary):
                        load_per_phase[f"P{phase}"][bus_index] += round(P_values[phase_index], 2)
                        load_per_phase[f"Q{phase}"][bus_index] += round(Q_values[phase_index], 2)

            loads_flag = self.loads.Next()
        
        return pd.DataFrame(load_per_phase)
    
    
    def get_pdelements(self):
        '''
        obtain all the power delivery elements
        '''
             
        element_activity_status = self.pdelements.First()
        element_data_list = []

        while element_activity_status:
            element_type = dss.CktElement.Name().lower().split('.')[0] 
            
            if element_type != 'capacitor':
                # capacitor is a shunt element
                
                if element_type == 'line':
                    each_element_data = {
                        'name': dss.Lines.Name(),
                        'element': element_type,
                        # from opendss manual -> length units = {none|mi|kft|km|m|ft|in|cm}
                        'length_unit': dss.Lines.Units(),
                        'z_matrix': self.__get_zmatrix(),
                        'length': dss.Lines.Length(),
                        'from_bus': dss.Lines.Bus1().split('.')[0],
                        'to_bus': dss.Lines.Bus2().split('.')[0],
                        'num_phases': dss.Lines.Phases(),
                        'is_switch': dss.Lines.IsSwitch(),
                        'is_open': (dss.CktElement.IsOpen(1, 0) or dss.CktElement.IsOpen(2, 0))
                    }      
                
                else:
                    # everything other than lines but not capacitors i.e. transformers, reactors etc.
                    each_element_data = {
                    'name': dss.CktElement.Name().split('.')[1],
                    'element': element_type,
                    'length_unit': 'NA',                
                    'z_matrix': np.zeros((3, 3)) + 1j * np.zeros((3, 3)),
                    'length': 0,
                    'from_bus': dss.CktElement.BusNames()[0].split('.')[0],
                    'to_bus': dss.CktElement.BusNames()[1].split('.')[0],
                    'num_phases': dss.CktElement.NumPhases(),
                    'is_switch': False,
                    'is_open': False
                    }
                    
                element_data_list.append(each_element_data)
            element_activity_status = self.pdelements.Next()
            
        return element_data_list


if __name__ == "__main__":
    dss_data = DSSpy(r"../../distribution_data/ieee9500_dss/Master-unbal-initial-config.dss")
    dss_data.get_transferred_loads()
    '''
        expectation from this script what data do we need:
    
    1. Linedata and lineparameters
    2. DERs and switches
    3. loaddata -> per phase load (P,Q) and their connected bus info
    4. Cycles 
    5. Normally closed switches (not immediate)
    '''
    
    breakpoint()
