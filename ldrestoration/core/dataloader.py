from __future__ import annotations
from typing import Any, TYPE_CHECKING
import json
from networkx.readwrite import json_graph
from pathlib import Path
import logging
import pandas as pd

if TYPE_CHECKING:
    import networkx as nx


from ldrestoration.utils.loggerconfig import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

from ldrestoration import DSSManager


class DataLoader:
    def __init__(self, data_folder_path: str = None, dss_file_path: str = None) -> None:
        """Class to loads the required data for the optimization model.
        Note: the files must be created via ldrestoration.dssparser.dssparser.DSSManager.saveparseddss module
        as this function expects the files to have the same name and type as saved by this module.

        Args:
            data_folder_path (str): Path or folder containing the required files
            dss_file_path (str): Path containing the master file of the OpenDSS model
        """

        self.data_folder_path = data_folder_path
        self.dss_file_path = dss_file_path

        if not (self.data_folder_path or self.dss_file_path):
            raise NotImplementedError(
                "Please provide either data_folder_path or dss_file_path to extract the data"
            )

        if self.data_folder_path:
            if not Path(self.data_folder_path).is_dir():
                msg = f"{self.data_folder_path} is not a valid directory"
                logger.error(msg)
                raise FileNotFoundError(
                    f"{self.data_folder_path} is not a valid directory. Please ensure that the path to the folder is correct."
                )

        if self.dss_file_path:
            if not Path(dss_file_path).is_file():
                msg = f"{self.dss_file_path} is not a valid dss file"
                logger.error(msg)
                raise FileNotFoundError(
                    f"{self.dss_file_path} is not a valid dss file. Please ensure that the path to the dss file is correct."
                )

    def from_dss(
        self,
        include_DERs: bool = False,
        DER_pf: float = 0.9,
        include_secondary_network: bool = False,
    ) -> dict[str, Any]:
        """Loads the required data for the optimization model from a dss master file.

        Returns:
            dictionary of overall data
        """

        if not self.dss_file_path:
            raise FileNotFoundError(
                f"dss_file_path = {self.dss_file_path} does not exist. Please call appropriate data model. Did you mean from_csv?"
            )

        dss_data = DSSManager(
            self.dss_file_path,
            include_DERs=include_DERs,
            DER_pf=DER_pf,
            include_secondary_network=include_secondary_network,
        )
        dss_data.parsedss()

        return {
            "network_graph": dss_data.network_graph,
            "network_tree": dss_data.network_tree,
            "loads": pd.DataFrame(dss_data.load_data),
            "DERs": pd.DataFrame(dss_data.DERs) if include_DERs else None,
            "normally_open_components": pd.DataFrame(
                dss_data.normally_open_components, columns=["normally_open_components"]
            ),
            "pdelements": pd.DataFrame(dss_data.pdelements_data),
            "circuit_data": dss_data.circuit_data,
        }

    def from_csv(self) -> dict[str, Any]:
        """Loads the required data for the optimization model from csv files.
        Note: the files must be created via ldrestoration.dssparser.dssparser.DSSManager.saveparseddss module
        as this function expects the files to have the same name and type as saved by this module.

        Returns:
            dictionary of overall data
        """
        if not self.data_folder_path:
            raise FileNotFoundError(
                f"data_folder_path = {self.data_folder_path} does not exist. Please call appropriate data model. Did you mean from_dss?"
            )

        return {
            "network_graph": self.load_network_graph(),
            "network_tree": self.load_network_tree(),
            "loads": self.load_network_loads(),
            "DERs": self.load_network_ders(),
            "normally_open_components": self.load_network_normally_open_components(),
            "pdelements": self.load_network_pdelements(),
            "circuit_data": self.load_circuit_data(),
        }

    def load_circuit_data(self) -> dict[str, Any]:
        """Loads the essential circuit data.

        Returns:
            dict[str, Any]: key is the metadata and value corresponds to the data for specific metadata
            eg. {"substation": "sourcebus"}
        """

        try:
            with open(Path(self.data_folder_path) / "circuit_data.json", "r") as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"The file {Path(self.data_folder_path) / 'circuit_data.json'} does not exist. "
                " Please make sure the folder contains files as created by the dataparser module."
            )

    def load_network_graph(self) -> nx.Graph:
        """Loads the networkx graph.

        Returns:
            nx.Graph: networkx graph of the model
        """

        try:
            with open(
                Path(self.data_folder_path) / "network_graph_data.json", "r"
            ) as file:
                network_graph_file = json.load(file)
                network_graph = json_graph.node_link_graph(network_graph_file)
            return network_graph
        except FileNotFoundError:
            raise FileNotFoundError(
                f"The file {Path(self.data_folder_path) / 'network_graph_data.json'} does not exist. "
                " Please make sure the folder contains files as created by the dataparser module."
            )

    def load_network_tree(self) -> nx.DiGraph:
        """Loads the networkx graph.

        Returns:
            nx.DiGraph: networkx tree of the model
        """

        try:
            with open(
                Path(self.data_folder_path) / "network_tree_data.json", "r"
            ) as file:
                network_tree_file = json.load(file)
                network_tree = json_graph.node_link_graph(network_tree_file)
            return network_tree
        except FileNotFoundError:
            raise FileNotFoundError(
                f"The file {Path(self.data_folder_path) / 'network_tree_data.json'} does not exist. "
                " Please make sure the folder contains files as created by the dataparser module."
            )

    def load_network_loads(self) -> pd.DataFrame:
        """Loads the network load i.e. active and reactive power demand.

        Returns:
            pd.DataFrame: DataFrame of overall load (active and reactive demand) of each node
        """

        try:
            loads = pd.read_csv(Path(self.data_folder_path) / "load_data.csv")
            return loads
        except FileNotFoundError:
            raise FileNotFoundError(
                f"The file {Path(self.data_folder_path) / 'load_data.csv'} does not exist."
                " Please make sure the folder contains files as created by the dataparser module."
            )

    def load_network_ders(self) -> pd.DataFrame | None:
        """Loads the DERs (distributed energy resources) if available.

        Returns:
            pd.DataFrame: DataFrame of DERs available in the network.
        """

        try:
            DERs = pd.read_csv(Path(self.data_folder_path) / "DERs.csv", delimiter=",")
            return DERs
        except pd.errors.EmptyDataError:
            logger.warning(
                "Empty file detected. This means that no DERs are detected in the base system."
            )
            return None
        except FileNotFoundError:
            logger.warning(
                "The DER file is missing. Please either pass include_DERs=True in DSSManager module or provide DER details."
                "The latter will be included in the future ..."
            )
            return None

    def load_network_normally_open_components(self) -> pd.DataFrame:
        """Loads the normally open components (tie switches and virtual DER switches) in the network, if available.

        Returns:
            pd.DataFrame: DataFrame of normally open components in the network.
        """

        try:
            normally_open_components = pd.read_csv(
                Path(self.data_folder_path) / "normally_open_components.csv"
            )
            return normally_open_components
        except FileNotFoundError:
            raise FileNotFoundError(
                f"The file {Path(self.data_folder_path) / 'normally_open_components.csv'} does not exist."
                " Please make sure the folder contains files as created by the dataparser module."
            )

    def load_network_pdelements(self) -> pd.DataFrame:
        """Loads the power delivery elements (line, transformer, switches, reactors) in the network.

        Returns:
            pd.DataFrame: DataFrame of pdelements in the network.
        """

        try:
            pdelements = pd.read_csv(
                Path(self.data_folder_path) / "pdelements_data.csv"
            )
            return pdelements
        except FileNotFoundError:
            raise FileNotFoundError(
                f"The file {Path(self.data_folder_path) / 'pdelements_data.csv'} does not exist."
                " Please make sure the folder contains files as created by the dataparser module."
            )

    def load_data(self) -> dict[str, Any]:
        """Loads the required data for the optimization model.
        Note: the files must be created via ldrestoration.dssparser.dssparser.DSSManager.saveparseddss module
        as this function expects the files to have the same name and type as saved by this module.

        Returns:
            [dict]: returns the dictionary of overall data
        """
        return dict(
            network_graph=self.load_network_graph(),
            network_tree=self.load_network_tree(),
            loads=self.load_network_loads(),
            DERs=self.load_network_ders(),
            normally_open_components=self.load_network_normally_open_components(),
            pdelements=self.load_network_pdelements(),
            circuit_data=self.load_circuit_data(),
        )
