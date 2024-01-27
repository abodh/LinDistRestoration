from __future__ import annotations

from folium import Map, CircleMarker, PolyLine
import pandas as pd
import plotly.graph_objects as go
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyomo.environ import ConcreteModel
    import networkx as nx

from ldrestoration.utils.networkalgorithms import loop_edges_to_tree_index

# plot color codes for distribution system plots
COLORCODES = {
    # key: circuit element type 
    # values: color, linewidth
    'transformer': ['green', 4],
    'switch': ['red', 4],
    'line': ['black', 1.5],
    'reactor': ['gray', 8]
}  

def plot_solution(model: ConcreteModel,
                  network_tree: nx.DiGraph,
                  network_graph: nx.Graph,
                  save_plot: bool = True,
                  filename: str = None) -> None:
    # Create a folium map centered at a specific location within the system
    distribution_map = Map(location=[network_tree.nodes[next(iter(network_tree.nodes))]['lat'],
                                        network_tree.nodes[next(iter(network_tree.nodes))]['lon']], 
                            zoom_start=20,
                            max_zoom=100)

    # circle markers as nodes of the system
    for node, data in network_tree.nodes(data=True):
        lat, lon = data['lat'], data['lon']
        
        if model.si[model.nodes.index(node)].value == 1:
            node_radius=2 
            node_color="black"
        else:
            node_radius=5 
            node_color="red"
        
        CircleMarker([lat, lon],
                     radius=1, 
                     color=node_color, 
                     tooltip=f"<span style='font-size:1.5em;'>{node}</span>").add_to(distribution_map)

    # These polylines connect nodes and distinguish their colors as per elements (transformer, line, switch)
    for from_node, to_node, data in network_graph.edges(data=True):
        source_data = network_tree.nodes[from_node]
        target_data = network_tree.nodes[to_node]            
        points = [[source_data['lat'], source_data['lon']], [target_data['lat'], target_data['lon']]]            
        
        
        try:
            edge_index = model.edges.index((from_node,to_node))
        except ValueError:
            edge_index = model.edges.index((from_node,to_node)[::-1])
            
        
        if model.xij[edge_index].value == 1:
            if data['is_switch']:
                colorcheck = "blue"
                weightcheck = 5
            else:
                colorcheck = "black"
                weightcheck = 1
        else:
            if data['is_switch']:
                colorcheck = "green"
                weightcheck = 5
            else:
                colorcheck = "red"
                weightcheck = 1
        
        PolyLine(points,
                 color="black",
                 weight=1, 
                 opacity=1,
                 tooltip=f"<span style='font-size:1.5em;'>{data['element'] if not data['is_switch'] else 'switch'}</span>").add_to(distribution_map)
    
    if save_plot:
        # Save the map to an HTML file
        distribution_map.save(f"{filename if filename else 'powerflow.html'}")
        

def plot_network_on_map(network_tree: nx.DiGraph,
                        network_graph: nx.Graph,
                        save_plot: bool = True,
                        filename: str = None) -> None:
    """Plot the distribution network on openstreet map if geographical coordinates are provided. 

    Args:
        network_tree (nx.DiGraph): Networkx tree. The tree should contain lat and lon data in each of the nodes
        network_graph (nx.Graph): Networkx graph. The graph should contain lat and lon data in each of the edges
        save_plot (bool, optional): Whether to save the plot or not. Defaults to True.
        filename (str, optional): filename of the plot. Defaults to None.
    """    
                                        
    # Create a folium map centered at a specific location within the system
    distribution_map = Map(location=[network_tree.nodes[next(iter(network_tree.nodes))]['lat'],
                                        network_tree.nodes[next(iter(network_tree.nodes))]['lon']], 
                            zoom_start=20,
                            max_zoom=50)

    # circle markers as nodes of the system
    for node, data in network_tree.nodes(data=True):
        lat, lon = data['lat'], data['lon']
        CircleMarker([lat, lon],
                     radius=3, 
                     color='blue', 
                     tooltip=f"<span style='font-size:1.5em;'>{node}</span>").add_to(distribution_map)

    # These polylines connect nodes and distinguish their colors as per elements (transformer, line, switch)
    for from_node, to_node, data in network_graph.edges(data=True):
        source_data = network_tree.nodes[from_node]
        target_data = network_tree.nodes[to_node]            
        points = [[source_data['lat'], source_data['lon']], [target_data['lat'], target_data['lon']]]            
        PolyLine(points,
                 color=COLORCODES[data['element'] if not data['is_switch'] else 'switch'][0],
                 weight=COLORCODES[data['element'] if not data['is_switch'] else 'switch'][1], 
                 opacity=1,
                 tooltip=f"<span style='font-size:1.5em;'>{data['element'] if not data['is_switch'] else 'switch'}</span>").add_to(distribution_map)
    
    if save_plot:
        # Save the map to an HTML file
        distribution_map.save(f"{filename if filename else 'distributionmap.html'}")

        
def plot_cartesian_network(cartesian_tree: nx.DiGraph,
                           cartesian_graph: nx.Graph,
                           save_plot: bool = True,
                           filename: str = None) -> None:
    """Plot the distribution network on a Cartesian coordinate system using Plotly.

    Args:
        cartesian_tree (nx.DiGraph): Networkx tree. The tree should contain X and Y data in each of the nodes
        cartesian_graph (nx.Graph): Networkx graph. The graph should contain X and Y data in each of the edges
        color_codes (dict): Color and linewidth codes for different circuit elements
        save_plot (bool, optional): Whether to save the plot or not. Defaults to True.
        filename (str, optional): Filename of the plot. Defaults to None.
    """    
    # Create a DataFrame for nodes
    nodes_df = pd.DataFrame([(node, data['lat'], data['lon']) 
                             for node, data in cartesian_tree.nodes(data=True)], columns=['node', 'X', 'Y'])

    # Create a DataFrame for edges
    edges_df = pd.DataFrame([(from_node, to_node, data['element'], data['is_switch']) 
                             for from_node, to_node, data in cartesian_graph.edges(data=True)], columns=['source',
                                                                                                         'target', 
                                                                                                         'element', 
                                                                                                         'is_switch'])

    # Create a scatter plot for nodes
    node_trace = go.Scatter(
        x=nodes_df['X'],
        y=nodes_df['Y'],
        mode='markers',
        marker=dict(
            size=10,
            color='blue'
        ),
        text=nodes_df['node'],
        hoverinfo='text'
    )

    # Create lines for edges
    edge_traces = []
    for _, edge in edges_df.iterrows():
        source_data = nodes_df[nodes_df['node'] == edge['source']]
        target_data = nodes_df[nodes_df['node'] == edge['target']]
        color = COLORCODES[edge['element'] if not edge['is_switch'] else 'switch'][0]
        linewidth = COLORCODES[edge['element'] if not edge['is_switch'] else 'switch'][1]
        edge_trace = go.Scatter(
            x=[source_data['X'].values[0], target_data['X'].values[0]],
            y=[source_data['Y'].values[0], target_data['Y'].values[0]],
            mode='lines',
            line=dict(
                color=color,
                width=linewidth
            ),
            hoverinfo='none'
        )
        edge_traces.append(edge_trace)

    # Create figure
    fig = go.Figure(data=[node_trace, *edge_traces])

    # Update layout
    fig.update_layout(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        showlegend=False
    )

    # Save or show the plot
    if save_plot:
        fig.write_html(f"{filename if filename else 'distributionmapxy.html'}")
    else:
        fig.show()