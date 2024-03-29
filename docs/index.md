# ![](logo.png)

!!! note
    The package is currently in a development stage and will have several code pushes in the next few weeks or so. Please use the current version carefully and stay updated with the latest version.

LinDistRestoration is a package to solve a 3-phase unbalanced distribution system restoration problem using mixed-integer linear programming. This package is intented for users interested in solving linear optimal power flow problem. The linear power flow model is extracted from [[1]](#ref1) and the base restoration models are based on [[2]](#ref2), [[3]](#ref3).

Currently, LinDistRestoration supports parsing OpenDSS raw files and creates the data files required to solve the restoration model. User can either provide a valid OpenDSS file or create separate data files, see `examples/dataparser/parsed_data_iee123` for specific data files and their structure. The base restoration model solves the 3-phase unbalanced optimal power flow to maximize the load restoration based on faulted lines. LinDistRestoration does not support fault location module and it is assumed that user provides the fault data as an input. 

## Getting started
Please follow this [installation](installation/installation.md){:target="_blank"} process and [examples](example/examples.md){:target="_blank"} to get started with LinDistRestoration.

## References
[](){#ref1}
[1] L. Gan and S. H. Low, "Convex relaxations and linear approximation for optimal power flow in multiphase radial networks," 2014 Power Systems Computation Conference, Warsaw, Poland, 2014, pp. 1-9.

[](){#ref2}
[2] A. Poudyal, S. Poudel and A. Dubey, "Risk-Based Active Distribution System Planning for Resilience Against Extreme Weather Events," in IEEE Transactions on Sustainable Energy, vol. 14, no. 2, pp. 1178-1192, April 2023.

[](){#ref3}
[3] S. Poudel, A. Dubey and K. P. Schneider, "A Generalized Framework for Service Restoration in a Resilient Power Distribution System," in IEEE Systems Journal, vol. 16, no. 1, pp. 252-263, March 2022.
