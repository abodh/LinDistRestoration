import networkx as nx
import logging

from ldrestoration.utils.decors import timethis
from ldrestoration.utils.loggerconfig import setup_logging
setup_logging()
logger = logging.getLogger(__name__)


@timethis
def network_cycles_basis(network_graph: nx.Graph)-> list[list[str]]:
    """returns cycle basis in a networkx graph i.e. fundamental cycles    
    Returns:
        cycles_edges (list[list[str]]) : list of cycles where each cycles contain the list of nodes in the cycle
    """    
    network_cycles = nx.cycle_basis(network_graph)
    
    cycles_edges = []
    for cycle in network_cycles:
        cycle_edges = []
        cycle_edges = [(cycle[i], cycle[(i+1) % len(cycle)]) for i in range(len(cycle))]
        cycles_edges.append(cycle_edges)   
    
    return cycles_edges

@timethis
def network_cycles_simple(network_graph: nx.Graph)-> list[list[str]]:
    """Returns simple cycles in a networkx graph. i.e. All the cycles in the network
    
    Returns:
        cycles_edges (list[list[str]]) : list of cycles where each cycles contain the list of nodes in the cycle
    """    
    network_cycles = nx.simple_cycles(network_graph)
    
    cycles_edges = []
    for cycle in network_cycles:
        cycle_edges = []
        cycle_edges = [(cycle[i], cycle[(i+1) % len(cycle)]) for i in range(len(cycle))]
        cycles_edges.append(cycle_edges)   
    
    return cycles_edges

@timethis
def edge_name_to_index(cycles: list[list[str]], 
                       edges: list[tuple[str,str]]) -> list[list[int]]:
    """Convert list of cycles with edges to their edge indices

    Args:
        cycles (list[list[str]]): list of cycle basis for the network 
        edges (list[tuple[str,str]]): list of edges in the network i.e.[(u1,v1), (u2, v2), ...]

    Raises:
        ValueError: if an edge from the cycle does not exist in the edge list

    Returns:
        loop_edge_idxs (list[list[int]]): list of cycles where each cycle is represented by their edge index 
    """    
    loop_edge_idxs = []
    for cycle in cycles:
        cycle_idxs = []
        for each_edge in cycle:
            try:
                cycle_idxs.append(edges.index(each_edge))
            except ValueError:
                try:
                    cycle_idxs.append(edges.index(each_edge[::-1]))
                except ValueError:
                    logger.debug(f"Could not find '{each_edge}' in the list of edges")
                    raise ValueError(f"{each_edge} is not a valid edge in model.edges.")            
        loop_edge_idxs.append(cycle_idxs)
    
    return loop_edge_idxs

@timethis
def associated_line_for_each_switch(graph: nx.Graph,
                                    switch_edge: tuple) -> dict[str,list[tuple[str,str]]]:
    
    """Access upstream and downstream switches associated with each non-switchable line

    Args:
        graph (nx.Graph): networkx graph of the network
        switch_edge (tuple): edge that is a switch

    Raises:
        ValueError: raised if an edge does not exist in the graph

    Returns:
        dict[str,list[tuple[str,str]]]: dictionary with key as an edge and value as list of associated switches 
    """    
    non_switchable_lines = {}
    visited_nodes = set(switch_edge)
    graph_edges = list(graph.edges())
    
    # recursive function to observe all the lines without switches
    def explore_all_lines(node: str, 
                          switch_associations: dict) -> None:
        """ explore upstream and downstream parts of the network for each node

        Args:
            node (str): node of interest upstream or downstream of which the traversal is to be made
            switch_associations (dict): associated switches with the current element (line or node)
            If the edge is a switch then key and value will be identical i.e. associated switch for a switch is itself 

        Raises:
            ValueError: raised if an edge does not exist in the graph
        """        
        
        
        # neighbors access adjacent nodes to the current node
        node_neighbors = graph.neighbors(node)
        
        # now go for every neighbor. This ensures we traverse upstream and downstream of the node
        for neighbor in node_neighbors:
            
            # if node is a parent node then form (node,neighbor) else form (neighbor,node)
            edge = (node, neighbor) if node == switch_edge[0] else (neighbor, node)
            
            # identify the index of that edge in the graph
            try: 
                graph_edges.index(edge) 
            except ValueError:
                try:
                    graph_edges.index(edge[::-1]) 
                    edge = edge[::-1]
                except ValueError:
                    logger.error(f"Please check the way edges are created in networkx as {edge} or {edge[::-1]} does not exist in your network graph or tree.")
                    raise ValueError(f"{edge} OR {edge[::-1]} does not exist in the network graph or tree.")
      
            # if the edge is a switch then we will skip as we are looking for adjacent non-switchable lines not switches
            if graph.edges[edge].get('is_switch'):
                if neighbor not in switch_associations:
                    switch_associations[neighbor] = [edge]
                else:
                    switch_associations[neighbor].append(edge)
                    
                # Skip switches
                continue
            
            # now we need to associate switches with the nonswitchable edges (lines)
            # if there is a non-switchable line then we append the switches
            if edge not in non_switchable_lines:
                non_switchable_lines[edge] = [neighbor]
            else:
                # the existence will only occur to complete a switch. 
                # So if this were to occur it will occur in exact 2nd trial. Confusing but think again :)
                non_switchable_lines[edge].append(neighbor)
                
            # call the recursive function until all the upstream and downstream nodes are visited for the current line
            # stop or continue if a switch is hit
            if neighbor not in visited_nodes:
                visited_nodes.add(neighbor)
                explore_all_lines(neighbor, switch_associations)
    
    # this will be switch associated with every node of the edge
    switch_associations = {}
    
    # explore lines for each node in the switch edge
    for node in switch_edge:
        explore_all_lines(node, switch_associations)
    
    return non_switchable_lines