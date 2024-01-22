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
def edge_name_to_index(cycles: list[list[str]], edges: list[tuple[str,str]]) -> list[list[int]]:
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