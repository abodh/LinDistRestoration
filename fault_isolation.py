# -*- coding: utf-8 -*-
"""
Created on: April 12, 2021
@author: Abodh Poudyal
"""

# Python program to print all paths from a source to destination. 
import numpy as np
import networkx as nx

def switch_identifier(fault_line_indices):
    # the first two are tie switches in 123 and rest are sectionalizers
    switches = np.matrix([[54, 94], [117, 123], [1, 125], [97, 124], [18, 116], [60, 119], [13, 121]])

    # Create a graph from the given edges
    nNodes = 125
    nEdges = 126
    Source = 125  # source node or substation node
    edges = np.loadtxt('data/edges.txt', dtype=int)
    n = (nNodes + 1, nNodes + 1)
    adjMatrix = np.zeros(n, dtype=int)

    # scan the arrays edge_u and edge_v
    for i in range(0, nEdges):
        u = edges[i, 0]  # from node
        v = edges[i, 1]  # to node
        adjMatrix[u, v] = 1  # adjacency matrix for the nodes and edges

    # Note: Only (u->v) connections are stored as 1 whereas (v->u) is not
    # u -> v <> v -> u

    # the main purpose of adjacency matrix is also to create a graph
    G = nx.from_numpy_matrix(adjMatrix)  # creates graph an adjacency numpy matrix
    Fault = fault_line_indices  # index of edge on which the fault has occurred

    # ways stores all possible simple paths from source to fault location
    # simple path -> path without forming any cycle
    ways = list(nx.all_simple_paths(G, source=Source, target=edges[Fault, 0]))

    '''
        * Now, at this point we know how many ways we have from a source to a faulted node
        * Ways store the number of different ways (nodes) for traversal from source node to faulted node
    '''
    vector = []
    for i in range(0, len(ways)):
        ls = len(ways[i])  # length of each of the ways from source to faulted node
        each_way = ways[i]  # each of the ways from source to faulted node
        isolate = []
        for m in range(0, ls - 1):

            # store variable stores a pair of nodes (each lines) starting from the end in each_way
            store = [each_way[ls - m - 1], each_way[ls - m - 2]]

            # check if the line in store variable is switch.
            # If yes, then we need to open that line for fault isolation.

            for k in range(0, len(switches)):
                if store[0] == switches[k, 0]:
                    if store[1] == switches[k, 1]:
                        # check if store(a,b) == switch (p,q)
                        isolate = switches[k]
                        vector.append(isolate)

                if store[0] == switches[k, 1]:
                    if store[1] == switches[k, 0]:
                        # check if store(a,b) == switch (q,p)
                        isolate = switches[k]
                        vector.append(isolate)

            if len(isolate) == 1:
                # breaks if a switch has been found
                break

    open_lines = []
    isolate_Fault = []
    vector = np.matrix(np.array(vector))

    for k in range(0, len(vector)):
        edge_index = np.where(np.all(vector[k] == edges, axis=1))
        open_lines.append(edge_index)
        each_way_2 = np.array(open_lines)
        isolate_Fault.append(each_way_2[k, 0, 0])

    isolate_Fault = set(isolate_Fault)

    # open_sw contains the indices of the edges that need to be isolated from the system
    open_sw = list(isolate_Fault)
    return open_sw
