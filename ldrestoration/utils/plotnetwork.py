from __future__ import annotations

from folium import Map, CircleMarker, PolyLine, Circle, IFrame, Popup, TileLayer
import pandas as pd
import plotly.graph_objects as go
from typing import TYPE_CHECKING
import branca
import numpy as np

if TYPE_CHECKING:
    from pyomo.environ import ConcreteModel
    import networkx as nx

''''
Note: the plot functions are currently in beta state. 
More changes will be introduced in future versions.
'''

# plot color codes for distribution system plots
COLORCODES = {
    # key: circuit element type 
    # values: color, linewidth
    'transformer': ['green', 4],
    'switch': ['red', 4],
    'line': ['black', 1.5],
    'reactor': ['gray', 8]
}  


def lat_lon_validator(lat: float, 
                     lon: float) -> None:
    """
    Validates latitude and longitude values.

    Args:
        lat (float): Latitude value to validate.
        lon (float): Longitude value to validate.
    """
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Latitude must be between -90 and 90 degrees, and "
                         "longitude must be between -180 and 180 degrees." 
                         "Please either provide valid lat lon coordinates or use plot_cartesian_network() instead.")

def plot_solution_map(model: ConcreteModel,
                      network_tree: nx.DiGraph,
                      network_graph: nx.Graph,
                      save_plot: bool = True,
                      filename: str = None,
                      background: str = None,
                      **kwargs) -> Map:
    
    """Plot the optimization solution in map with colored representation for each of the components
    
    Args:
        model (ConcreteModel): Solved Pyomo concrete model 
        network_tree (nx.DiGraph): networkx tree representation of the system
        network_graph (nx.Graph): networkx graph representation of the system
        save_plot (bool): whether to save the plot or not
        filename (str): filename of the plot file to save       
        background (str): Plot background. Can either choose "white" or "map"
        **kwargs: other keyword arguments applicable to folium imports
    
    Returns:
        Map:  The customized folium map. Users can customize the map as preferred.  
    """    
    
    # validate if the lat lon coordinates are valid. We only check for the first coordinate 
    lat_lon_validator(network_tree.nodes[next(iter(network_tree.nodes))]['lat'],
                     network_tree.nodes[next(iter(network_tree.nodes))]['lon'])
    
    if background not in ["white", "map", None]:
        raise ValueError("background can either be white, map, or None (defaults to map). Please provide a valid background.")
    
    if background == "white":
        white_tile = branca.utilities.image_to_url([[1, 1], [1, 1]])
        
        # Create a folium map centered at a specific location within the system
        distribution_map = Map(location=[network_tree.nodes[next(iter(network_tree.nodes))]['lat'],
                                        network_tree.nodes[next(iter(network_tree.nodes))]['lon']], 
                            tiles=white_tile, attr="white tile",
                            zoom_start=13,
                            max_zoom=100,
                            **kwargs)
    else:
        # if not white then we default to map for "map" or None background        
        # Create a folium map centered at a specific location within the system
        distribution_map = Map(location=[network_tree.nodes[next(iter(network_tree.nodes))]['lat'],
                                        network_tree.nodes[next(iter(network_tree.nodes))]['lon']],
                            zoom_start=13,
                            max_zoom=100,
                            # crs="EPSG4326",
                            **kwargs)        
    
    fault_indices = []
    for fault in model.faults:
        try:
            fault_index = model.edges.index(fault) 
        except ValueError:
            fault_index = model.edges.index(fault[::-1])
        fault_indices.append(fault_index)
        
            
    powerflow = {index: abs(round(model.Pija[edge].value + model.Pijb[edge].value + model.Pijc[edge].value, 2)) for index, edge in enumerate(model.x_ij)}
    min_flow = min(powerflow.values())
    max_flow = max(powerflow.values())
    widthmin = 1
    widthmax = 20 
    
    # circle markers as nodes of the system
    for node, data in network_tree.nodes(data=True):
        lat, lon = data['lat'], data['lon']        
        node_index = model.nodes.index(node)
        if round(model.si[node_index].value) == 1:
            node_radius=2 
            node_color="black"
        else:
            node_radius=2 
            node_color="gray"
        
        popup_content = f"<span style='font-size:1.5em;'><strong>Name</strong>: {node}<br> \
                   <strong>Va</strong>: {round(np.sqrt(model.Via[node_index].value), 3) if model.Via[node_index].value is not None else 0.95}<br> \
                       <strong>Vb</strong>: {round(np.sqrt(model.Vib[node_index].value), 3) if model.Vib[node_index].value is not None else 0.95}<br> \
                           <strong>Vc</strong>: {round(np.sqrt(model.Vic[node_index].value), 3) if model.Vic[node_index].value is not None else 0.95}<br> \
                               <strong>pickup status</strong>: {model.si[node_index].value}<br> \
                                   <strong>energization status</strong>: {model.vi[node_index].value} </span>"
        
        iframe = IFrame(popup_content)
        popup = Popup(iframe,
                      min_width=300,
                      max_width=300
                      )
        
        Circle([lat, lon],
               radius=node_radius, 
               color=node_color, 
               fill_color=node_color,
               fill_opacity = 0.7,
               opacity=0.7, 
               tooltip=f"<span style='font-size:1.5em;'><strong>Name</strong>: {node} </span>",
               popup=popup,
               **kwargs).add_to(distribution_map)

    # These polylines connect nodes and distinguish their colors as per elements (transformer, line, switch)
    for from_node, to_node, data in network_graph.edges(data=True):
        source_data = network_tree.nodes[from_node]
        target_data = network_tree.nodes[to_node]            
        points = [[source_data['lat'], source_data['lon']], 
                  [target_data['lat'], target_data['lon']]]            
        
        try:
            edge_index = model.edges.index((from_node,to_node))
        except ValueError:
            edge_index = model.edges.index((to_node, from_node))
        
        if edge_index in model.virtual_switch_indices:
            if from_node == 'sourcebus':
                
                popup_content = f"<span style='font-size:1.5em;'><strong>Name</strong>: {model.DERs[model.DERs['connected_bus'] == to_node]['name'].item()} <br> \
                                 <strong>Bus</strong>: {to_node} <br> \
                                     <strong>Rated kW</strong>: {model.DERs[model.DERs['connected_bus'] == to_node]['kW_rated'].item()} kW <br>\
                                         <strong>Generated kW</strong>: {powerflow[edge_index]} kW </span>"
                                     
        
                iframe = IFrame(popup_content)
                popup = Popup(iframe,
                              min_width=300,
                              max_width=300
                              )
                
                Circle([source_data['lat'], source_data['lon']],
                       radius=100, 
                       color='green', 
                       fill_color='green',
                       fill_opacity = 0.7,
                       opacity=0.7,
                       popup=popup,
                       tooltip=f"<span style='font-size:1.5em;'><strong>DER</strong>: {model.DERs[model.DERs['connected_bus'] == to_node]['name'].item()} </span>").add_to(distribution_map)
            else:
                
                popup_content = f"<span style='font-size:1.5em;'><strong>Name</strong>: {model.DERs[model.DERs['connected_bus'] == from_node]['name'].item()} <br> \
                                 <strong>Bus</strong>: {from_node} <br> \
                                     <strong>Rated kW</strong>: {model.DERs[model.DERs['connected_bus'] == from_node]['kW_rated'].item()} kW <br>\
                                         <strong>Generated kW</strong>: {powerflow[edge_index]} kW </span>"
        
                iframe = IFrame(popup_content)
                popup = Popup(iframe,
                              min_width=300,
                              max_width=300
                              )
                
                Circle([source_data['lat'], source_data['lon']],
                             radius=100, 
                             color='green',  
                             fill_color='green',
                             fill_opacity = 0.7,
                             opacity=0.7,
                             popup=popup,
                             tooltip=f"<span style='font-size:1.5em;'><strong>DER</strong>: {model.DERs[model.DERs['connected_bus'] == from_node]['name'].item()} </span>",
                             **kwargs).add_to(distribution_map)
            
            # place a marker and continue since we do not want virtual edges to display
            continue

        if round(model.xij[edge_index].value) == 1:
            if data['is_switch']:
                # if normally open switch is closed then green
                # if sectionalizer is closed then gray
                colorcheck = "green" if data['is_open'] else "blue"
                weightcheck = 5
            else:        
                if powerflow[edge_index] > 0: 
                    # normal line
                    colorcheck = "black"
                    weightcheck = (widthmax - widthmin) * ((powerflow[edge_index] - min_flow)/ (max_flow - min_flow)) + widthmin
                else:
                    colorcheck = "gray"
                    weightcheck = widthmin 
        else:
            # condition when line status is open
            if edge_index in fault_indices:    
                # only sectionalizers are faulted no tie so we do not even check if it is switch          
                # every faulted line is red  
                colorcheck = "red"
                weightcheck = 5
            else:
                if data['is_switch']:
                    # if non faulted line is opened then either there is no flow or its tie switch
                    colorcheck = "pink" if data['is_open'] else "red"
                    weightcheck = 5
                else:
                    colorcheck = "gray"
                    weightcheck = widthmin
        
        
        # put in the contents for fault logic here
        # if fault then color is red

        popup_content = f"<span style='font-size:1.5em;'><strong>Name</strong>: {data['name']}<br> \
            <strong>Pa</strong>: {abs(round(model.Pija[edge_index].value, 2))} kW<br> \
                     <strong>Pb</strong>: {abs(round(model.Pijb[edge_index].value, 2))} kW<br> \
                         <strong>Pc</strong>: {abs(round(model.Pijc[edge_index].value, 2))} kW<br> \
                             <strong>element</strong>: {data['element'] if not data['is_switch'] else 'switch'} <br>\
                                 <strong>connectivity status</strong>: {round(model.xij[edge_index].value)}</span>"
        
        iframe = IFrame(popup_content)
        popup = Popup(iframe,
                        min_width=300,
                        max_width=300
                        )
        PolyLine(points,
                 color=colorcheck,
                 weight=weightcheck, 
                 opacity=1,
                 popup=popup,
                 tooltip=f"<span style='font-size:1.5em;'><strong>Name</strong>: {data['name']} </span>",
                 **kwargs).add_to(distribution_map)
                 
    if save_plot:
        # Save the map to an HTML file
        distribution_map.save(f"{filename if filename else 'powerflow.html'}")
        
    return distribution_map
        

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
                            max_zoom=80)

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

    fig = go.Figure(data=[node_trace, *edge_traces])
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