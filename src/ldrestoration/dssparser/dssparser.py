from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from altdss import altdss as dss
from ldrestoration.utils.loggerconfig import logger
from networkx.readwrite import json_graph

from ldrestoration.dssparser import (
    BusHandler,
    LoadHandler,
    NetworkHandler,
    PDElementHandler,
    TransformerHandler,
)
from ldrestoration.utils.decors import timethis


class DSSManager:
    """DSSManager is the primary module to parse the OpenDSS data. It manages all the components, including but not limited to, loads, generators, pdelements, transformers etc.
    Each of the components' data structure is managed by their respective handlers and can be accessed individually, if required.

    Args:
        dssfile (str): path of the dss master file (currently only supports OpenDSS files)
        include_DERs (bool, optional): Check whether to include DERs or not. Defaults to True.
        DER_pf (float, optional): Constant power factor of DERs. Defaults to 0.9.
        include_secondary_network (bool, optional): Check whether to include secondary network or not. Defaults to False.

    Examples:
        The only required argument is the OpenDSS master file. We assume that the master file compiles all other OpenDSS files.
        The DSSManager class is initiated first and a method parse_dss() will then parse the overall data.
        >>> dataobj = DSSManager("ieee123master.dss", include_DERs=True)
        >>> dataobj.parse_dss()

    """

    def __init__(
        self,
        dssfile: Path | str,
        include_DERs: bool = False,
        DER_pf: float = 0.95,
        include_secondary_network: bool = False,
    ) -> None:
        """Initialize a DSSManager instance. This instance manages all the components in the distribution system.

        Args:
            dssfile (Path | str): path of the dss master file
            include_DERs (bool, optional): Check whether to include DERs or not. Defaults to False.
            DER_pf (float, optional): Constant power factor of DERs. Defaults to 0.95.
            include_secondary_network (bool, optional): Check whether to include secondary network or not. Defaults to False.
        """
        msg = "Initializing DSSManager"
        logger.info(msg)

        if isinstance(dssfile, str):
            self.dssfile = Path(dssfile)

        dss.Settings.AllowChangeDir = False
        self.dss = dss.NewContext()
        self.dss.ClearAll()
        self.dss(f'Compile "{self.dssfile}"')

        # initialize other attributes
        self.include_DERs = include_DERs  # variable to check whether to include DERs or not
        self.DER_pf = DER_pf  # constant power factor of DERs
        self.include_secondary_network = include_secondary_network  # check whether to include secondary or not

        self.DERs = None  # variable to store information on DERs if included
        self.pv_systems = None
        self.circuit_data = {}  # store circuit metadata such as source bus, base voltage, etc

        # initialize parsing process variables and handlers
        self._initialize()

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
        return self.dss.Vsource.BasekV()[0]

    @property
    def source(self) -> str:
        """source bus of the circuit.

        Returns:
            str: returns the source bus of the circuit
        """
        return self.dss.Vsource.Name[0]

    @timethis
    def _initialize(self) -> None:
        """
        Initialize user-based preferences as well as DSS handlers (i.e. load, transformer, pdelements, and network)
        """
        # if DERs are to be included then include virtual switches for DERs
        if self.include_DERs:
            self._initializeDERs()
            self._initialize_PV()
            msg_initialization = f"DERs virtual switches have been added successfully. We assume a constant power factor of DERs; DERs power factor = {self.DER_pf}"
            logger.info(msg_initialization)
        else:
            msg = f"DERs virtual switches are excluded as include_DERs = {self.include_DERs}."
            logger.info(msg)

        # initialize DSS handlers
        self._initialize_dsshandlers()

        # initialize data variables (these will be dynamically updated through handlers)
        self.bus_data = None
        self.transformer_data = None
        self.pdelements_data = None
        self.network_graph = None
        self.network_tree = None
        self.normally_open_components = None
        self.load_data = None

    @timethis
    def _initializeDERs(self) -> None:
        """
        Include or exclude virtual switches for DERs based on DER inclusion flag
        """
        self.DERs = []
        generator_flag = self.dss.Generators.First()
        while generator_flag:
            self.dss.Text.Command(
                "New Line.{virtual_DERswitch} phases=3 bus1={source_bus} bus2={gen_bus} switch=True r1=0.001 r0=0.001 x1=0.001 x0=0.001 C1=0 C0=0 length=0.001".format(
                    virtual_DERswitch=self.dss.Generators.Name(),
                    source_bus=self.source,
                    gen_bus=self.dss.Generators.Bus1(),
                )
            )
            self.DERs.append(
                {
                    "name": self.dss.Generators.Name(),
                    "kW_rated": round(self.dss.Generators.kVARated() * self.DER_pf, 2),
                    "connected_bus": self.dss.Generators.Bus1(),
                    "phases": self.dss.Generators.Phases(),
                }
            )

            generator_flag = self.dss.Generators.Next()

        # we also need to ensure that these switches are open as they are virtual switches
        for each_DERs in self.DERs:
            self.dss.Text.Command(f"Open Line.{each_DERs['name']}")

        self.dss.Solution.Solve()

    @timethis
    def _initialize_PV(self) -> None:
        """
        Include PV Systems from the dss data
        """
        self.pv_systems = []
        pv_flag = self.dss.PVsystems.First()
        while pv_flag:
            self.pv_systems.append(
                {
                    "name": self.dss.PVsystems.Name(),
                    "kW_rated": round(self.dss.PVsystems.kVARated() * self.dss.PVsystems.pf(), 2),
                    "connected_bus": self.dss.CktElement.BusNames()[0],
                    "phases": self.dss.CktElement.NumPhases(),
                }
            )

            pv_flag = self.dss.PVsystems.Next()

    @timethis
    def _initialize_dsshandlers(self) -> None:
        """Initialize all the DSS Handlers"""

        # bus_handler is currently not being used here but kept here for future usage
        self.bus_handler = BusHandler(self.dss)
        self.transformer_handler = TransformerHandler(self.dss)
        self.pdelement_handler = PDElementHandler(self.dss)
        self.network_handler = NetworkHandler(self.dss, pdelement_handler=self.pdelement_handler)

        if self.include_secondary_network:
            logger.info("Considering entire system including secondary networks")
            self.load_handler = LoadHandler(self.dss, include_secondary_network=self.include_secondary_network)
        else:
            # if primary loads are to be referred then we must pass network and transformer handlers
            logger.info("Considering primary networks and aggregating loads by referring them to the primary node")
            self.load_handler = LoadHandler(
                self.dss,
                include_secondary_network=self.include_secondary_network,
                network_handler=self.network_handler,
                transformer_handler=self.transformer_handler,
            )
        logger.info(f'Successfully instantiated required handlers from "{self.dssfile}"')

    @timethis
    def parsedss(self) -> None:
        """Parse required data from the handlers to respective class variables"""
        self.bus_data = self.bus_handler.get_buses()
        self.transformer_data = self.transformer_handler.get_transformers()
        self.pdelements_data = self.pdelement_handler.get_pdelements()
        self.network_graph, self.network_tree, self.normally_open_components = self.network_handler.network_topology()
        self.load_data = self.load_handler.get_loads()

        if not self.include_secondary_network:
            logger.info(f"Excluding secondaries from final tree, graph configurations, and pdelements.")
            self.network_tree.remove_nodes_from(self.load_handler.downstream_nodes_from_primary)
            self.network_graph.remove_nodes_from(self.load_handler.downstream_nodes_from_primary)
            self.pdelements_data = [
                items
                for items in self.pdelements_data
                if items["from_bus"] not in self.load_handler.downstream_nodes_from_primary
                and items["to_bus"] not in self.load_handler.downstream_nodes_from_primary
            ]
        logger.info(f"Successfully parsed the required data from all handlers.")

        # the networkx data is saved as a serialized JSON
        self.network_graph_data = json_graph.node_link_data(self.network_graph)
        self.network_tree_data = json_graph.node_link_data(self.network_tree)

        # parse additional circuit data
        # add more as required in the future ...
        self.circuit_data = {
            "substation": self.source,
            "basekV_LL_circuit": self.basekV_LL,
        }

    @timethis
    def saveparseddss(self, folder_name: str = f"parsed_data", folder_exist_ok: bool = False) -> None:
        """Saves the parsed data from all the handlers

        Args:
            folder_name (str, optional): Name of the folder to save the data in. Defaults to "dssdatatocsv"_<current system date>.
            folder_exist_ok (bool, optional): Boolean to check if folder rewrite is ok. Defaults to False.
        """

        # check if parsedss is run before saving these files
        if self.bus_data is None:
            logger.error(
                "Please run DSSManager.parsedss() to parse the data and then run this function to save the files."
            )
            raise NotImplementedError(
                f"Data variables are empty. You must run {__name__}.DSSManager.parsedss() to extract the data before saving them."
            )

        # check if the path already exists. This prevent overwrite
        try:
            Path(folder_name).mkdir(parents=True, exist_ok=folder_exist_ok)
        except FileExistsError:
            logger.error(
                "The folder already exists and the module is attempting to rewrite the data in the folder. Either provide a path in <folder_name> or mention <folder_exist_ok=True> to rewrite the existing files."
            )
            raise FileExistsError("The folder or files already exist. Please provide a non-existent path.")

        # save all the data in the new folder
        # the non-networkx data are all saved as dataframe in csv
        pd.DataFrame(self.bus_data).to_csv(f"{folder_name}/bus_data.csv", index=False)
        pd.DataFrame(self.transformer_data).to_csv(f"{folder_name}/transformer_data.csv", index=False)
        pd.DataFrame(self.pdelements_data).to_csv(f"{folder_name}/pdelements_data.csv", index=False)
        pd.DataFrame(self.load_data).to_csv(f"{folder_name}/load_data.csv", index=False)
        pd.DataFrame(self.normally_open_components, columns=["normally_open_components"]).to_csv(
            f"{folder_name}/normally_open_components.csv", index=False
        )

        if self.DERs is not None:
            pd.DataFrame(self.DERs).to_csv(f"{folder_name}/DERs.csv", index=False)
            pd.DataFrame(self.pv_systems).to_csv(f"{folder_name}/pv_systems.csv", index=False)

        with open(f"{folder_name}/network_graph_data.json", "w") as file:
            json.dump(self.network_graph_data, file)

        with open(f"{folder_name}/network_tree_data.json", "w") as file:
            json.dump(self.network_tree_data, file)

        # save the circuit data as json
        with open(f"{folder_name}/circuit_data.json", "w") as file:
            json.dump(self.circuit_data, file)

        logger.info("Successfully saved required files.")


@timethis
def main() -> None:
    dss_data = DSSManager(
        r"../../examples/test_cases/ieee9500_dss/Master-unbal-initial-config.dss",
        include_DERs=False,
        include_secondary_network=False,
    )
    dss_data.parsedss()
    dss_data.saveparseddss()


if __name__ == "__main__":
    main()
