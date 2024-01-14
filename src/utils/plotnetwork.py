from folium import Map, CircleMarker, PolyLine

# plot color codes for distribution system plots
COLORCODES = {
    # key: circuit element type 
    # values: color, linewidth
    'transformer': ['green', 4],
    'switch': ['red', 4],
    'line': ['black', 1.5],
    'reactor': ['gray', 8]
}  

def plot_network(network_tree,
                 network_graph,
                 plot_network=True,
                 filename=None):
    
    if plot_network:
                                
        # Create a folium map centered at a specific location within the system
        distribution_map = Map(location=[network_tree.nodes[next(iter(network_tree.nodes))]['lat'],
                                            network_tree.nodes[next(iter(network_tree.nodes))]['lon']], 
                                zoom_start=20,
                                max_zoom=50)

        # circle markers ars nodes of the system
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
        
        
        # Save the map to an HTML file
        distribution_map.save(f"{filename if filename else 'distributionmap.html'}")