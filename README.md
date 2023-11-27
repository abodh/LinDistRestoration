# power-distribution-restoration
A simple power distribution restoration method implemented on the IEEE-123 bus system using Python and Pyomo. The restoration model works both on single or multi-fault conditions. This implementation is based on the following two works:
1. S. Poudel, A. Dubey and K. P. Schneider, "A Generalized Framework for Service Restoration in a Resilient Power Distribution System," in IEEE Systems Journal, vol. 16, no. 1, pp. 252-263, March 2022
2. A. Poudyal, S. Poudel and A. Dubey, "Risk-Based Active Distribution System Planning for Resilience Against Extreme Weather Events," in IEEE Transactions on Sustainable Energy, vol. 14, no. 2, pp. 1178-1192, April 2023

# Data
The data required for the optimization model is stored as a text file in the data folder. Users do not need to access any external files for this work. However, the data folder must be updated, or additional scripts should be written to handle cases other than the IEEE-123 bus test case. The following text files can be extracted from any distribution system simulator (DSS) as required:
1. line_config: line configuration of each of the edges (col1: from node, col2: to node, col3: line length (in ft), col4: line configuration, col5: unused garbage value
2. edges: edge connection for the network (col1: from node, col2: to node)
3. load_data (per phase load data. Odd columns represent active power and even columns represent reactive power for each of the phases)

There is an additional text file named ```cycles.txt```. This is an externally generated file and is not readily available through DSS. These are all simple cycles in a network that can occur due to tie switches or distributed generators (DGs) modeled with virtual switches. For more information, please refer to the literature mentioned above. There are only 3 cycles in a base IEEE-123 bus system. However, as shown in the file, we have included 3 DGs in this network (node 95, 115, 122), which creates additional cycles. Each cycle contains the edge indices associated with the cycle.

The Z matrix associated with each edge is also stored in the ```Zmatrix<>.py``` file associated with each phase. This can be extracted from the DSS files of the test case as well for future works. 

# Modules
There are 3 modules in this work. The primary one is ```main.py```, where faulted edges are defined for fault isolation and restoration. This module calls ```fault_isolation.py``` module to identify the switches that need to be toggled to isolate the faulted section in the network. This process is known as fault isolation. Finally, the restoration model is called for reconfiguration and DG-assisted restoration to restore the remaining parts of the network that were unserved during the fault isolation process. The entire problem is modeled as a mixed integer linear programming problem where the power flow is modeled using _LinDistFlow_ model. 

# Note
This is a simple fault isolation and restoration model implemented on the IEEE-123 bus for demonstration purposes. Please feel free modify the content and submit a pull request with additional innovations. The current version has been tested and works well for single- or multi-fault conditions.

