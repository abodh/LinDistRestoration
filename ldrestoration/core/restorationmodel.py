from __future__ import annotations

from typing import Union, Any, TYPE_CHECKING
from functools import cache
import logging
import numpy as np
import pandas as pd
from pyomo.environ import (
    ConcreteModel,
    Var,
    Objective,
    Constraint,
    RangeSet,
    ConstraintList,
    Binary,
    Reals,
    maximize,
    SolverFactory,
    value,
)
from pyomo.opt import SolverStatus, TerminationCondition

if TYPE_CHECKING:
    from pyomo.opt.results.results_ import SolverResults

from ldrestoration.utils.decors import timethis
from ldrestoration.core.dataloader import DataLoader
from ldrestoration.utils.networkalgorithms import (
    network_cycles_basis,
    loop_edges_to_tree_index,
    associated_line_for_each_switch,
)

from ldrestoration.utils.loggerconfig import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# the multiplier here means that the big M for each branch will be 10 times the max substation flow
# for voltage 5 is enough since distribution systems voltage ranges from 0.95 to 1.05 per unit
BIG_M_POWER_MULTIPLIER = 10
BIG_M_VOLTAGE_MULTIPLIER = 5
KW_TO_MW_FACTOR = 1000


# exceptions
class missing_kVbase_error(Exception):
    "Raise exception when the line to line kV Base is not passed as an argument"


class RestorationModel:
    def __init__(self, data: DataLoader, faults: list[tuple] = None) -> None:
        """LinDistRestoration model

        Args:
            data (DataLoader): instance of DataLoader class to load all of the required data
            faults (list[str]): list of line element edges with faults in (u,v) format

        Raises:
            FileNotFoundError: exception is raised when one or more files are missing
        """
        # assign attributes from the inputs
        self.data = data.load_data()
        if faults is None:
            self.faults = []
        else:
            self.faults = faults

        # create attributes for the data
        try:
            self.network_graph = self.data["network_graph"]
            self.network_tree = self.data["network_tree"]
            self.loads = self.data["loads"]
            self.pdelements = self.data["pdelements"]
            self.normally_open_components = self.data["normally_open_components"]
            self.circuit_data = self.data["circuit_data"]
        except KeyError:
            logger.debug(
                "One or more files are missing. Please include all the essential files."
            )
            raise FileNotFoundError(
                "One or more files are missing. Please make sure you included the following files: "
                "network graph, networkx tree, pdelements, loads, normally open components, DERs [Optional]"
            )

        # This is a personal taste. You can either use Concrete Model or Abstract model.
        # Concrete model is usually preferred due to visual aesthetics and its pythonic similarity
        self.model = ConcreteModel()
        self._constraints_list = None

        # the model data and variables are initialized within the class
        self._initialize_model_data()

        logger.info("Restoration model is successfully initialized.")

    @property
    def DERs(self) -> Union[None, pd.DataFrame]:
        """DERs data. Returns the data if DERs are included else None

        Returns:
            Union[None|pd.DataFrame]: Either DER data in DataFrame format or None
        """
        # whether DERs exist or not
        if "DERs" in self.data:
            return self.data["DERs"]
        else:
            # DERs are not included in this case
            return None

    @property
    def all_constraints(self) -> list[ConstraintList]:
        """list of all constraints in the existing model

        Returns:
            list[ConstraintList]: list of ConstraintList of the initialized model
        """
        if self._constraints_list is None:
            logger.error(
                "Constraints are not created yet. You need to execute one of the constraint sets before accessing this attribute."
            )
            raise ValueError(
                "Constraints are not created yet. Please create the constraints for your model before accessing them."
            )
        else:
            return self._constraints_list

    @property
    @cache
    def edge_indices_in_tree(self) -> dict[tuple, int]:
        """Identify the edge indices of the all the pdelements in the network tree. Since networkx does not consider the ways edges
        are added in the network for undirected graphs (u,v and v,u), we need to ensure we get  proper edge index for each pdelements.

        Raises:
            ValueError: if edge is not identified as either uv or vu in the network edges

        Returns:
            dict[tuple,int]: dictionary with key as edge in the form (u,v) and value as its index in the network edge
        """

        edge_index_map = {edge: index for index, edge in enumerate(self.model.edges)}
        edge_indices = {}
        for _, each_line in self.pdelements.iterrows():
            try:
                edge_indices[(each_line["from_bus"], each_line["to_bus"])] = (
                    edge_index_map[(each_line["from_bus"], each_line["to_bus"])]
                )
            except KeyError:
                try:
                    edge_indices[(each_line["from_bus"], each_line["to_bus"])] = (
                        edge_index_map[(each_line["to_bus"], each_line["from_bus"])]
                    )
                except KeyError:
                    logger.error(
                        f"Please check the way edges are created in networkx as neither {(each_line['from_bus'], each_line['to_bus'])} or {(each_line['from_bus'], each_line['to_bus'])[::-1]} exist in your network graph or tree."
                    )
                    raise ValueError(
                        f"{(each_line['from_bus'], each_line['to_bus'])} OR {(each_line['to_bus'], each_line['from_bus'])} does not exist in the network graph or tree."
                    )
        return edge_indices

    @property
    @cache
    def node_indices_in_tree(self) -> dict[str, int]:
        """Identify the indices of each nodes in the tree using hash map

        Returns:
            dict[str,int]: dictionary with key as node name and value as node index in the tree
        """
        return {node: index for index, node in enumerate(self.model.nodes)}

    @property
    @cache
    def demand_node_indices_in_tree(self) -> dict[int, int]:
        """Map the demand node of the demand dataframe with the node indices of the tree.

        Raises:
            ValueError: if node is not identified in the network nodes

        Returns:
            dict[int,int]: dictionary with key as node index and value as its corresponding index in the demand dataframe
        """
        node_indices = {}
        for demand_index, each_row in self.model.demand.iterrows():
            try:
                node_indices[self.node_indices_in_tree[each_row["bus"]]] = demand_index
            except ValueError:
                logger.error(
                    f"Please check the nodes in the network graph/tree as {each_row['name']} does not exist."
                )
                raise ValueError(
                    f"{each_row['name']} does not exist in the network graph or tree."
                )

        return node_indices

    @property
    @cache
    def lines_to_switch_mapper(self) -> dict[str, list[tuple[str, str]]]:
        """Maps line elements to corresponding upstream and downstream switches.

        Returns:
            dict[str, list[tuple[str,str]]]: dict with key as line element and value as list of switches associated with the element
        """
        line_to_switch = {}
        sectionalizers = list(
            zip(
                self.model.sectionalizing_switches["from_bus"],
                self.model.sectionalizing_switches["to_bus"],
            )
        )
        all_switches = sectionalizers + self.normally_open_tuples

        for switch_edge in all_switches:
            element_to_switch_mapper = associated_line_for_each_switch(
                self.network_graph, switch_edge
            )
            for line, _ in element_to_switch_mapper.items():
                if line not in line_to_switch:
                    line_to_switch[line] = [switch_edge]
                else:
                    line_to_switch[line].append(switch_edge)

        return line_to_switch

    @property
    @cache
    def sectionalizing_switch_indices(self) -> list[int]:
        """List of edge indices (as per network) for all the sectionalizing switches

        Returns:
            list[int]: returns list of sectionalizing switch indices as per network edge indices
        """

        sectionalizers = [
            self.edge_indices_in_tree[(each_row["from_bus"], each_row["to_bus"])]
            for _, each_row in self.model.sectionalizing_switches.iterrows()
        ]
        logger.info(f"Sectionalizing switch indices: {sectionalizers}")

        return sectionalizers

    @property
    @cache
    def tie_switch_indices(self) -> list[int]:
        """List of edge indices (as per network) for all the tie switches

        Returns:
            list[int]: returns list of tie switch indices as per network edge indices
        """

        tie_switches = [
            self.edge_indices_in_tree[(each_row["from_bus"], each_row["to_bus"])]
            for _, each_row in self.model.tie_switches.iterrows()
        ]
        logger.info(f"tie switch indices: {tie_switches}")

        return tie_switches

    @property
    @cache
    def virtual_switch_indices(self) -> list[int]:
        """List of edge indices (as per network) for all the virtual switches

        Returns:
            list[int]: returns list of virtual switch indices as per network edge indices
        """

        if self.model.virtual_switches is not None:
            virtual_switches = [
                self.edge_indices_in_tree[(each_row["from_bus"], each_row["to_bus"])]
                for _, each_row in self.model.virtual_switches.iterrows()
            ]
            logger.info(f"virtual switch indices: {virtual_switches}")
        else:
            virtual_switches = []
            logger.info(f"virtual switches do not exist as there are no DERs.")

        return virtual_switches

    @property
    @cache
    def all_switch_indices(self) -> set[int]:
        """Set of all switch indices as per network edge (tie + sectionalizing + virtual (if DERs exist))

        Returns:
            set[int]: returns set of all switch indices as per network edge indices
        """

        return set(self.sectionalizing_switch_indices).union(
            set(self.tie_switch_indices), set(self.virtual_switch_indices)
        )

    @timethis
    def _initialize_model_data(self) -> None:
        """Initialize the restoration model by preparing the model and required data structure"""

        # system pdelements
        self.model.pdelements = self.pdelements

        # normally closed sectionalizing switches
        self.model.sectionalizing_switches = self.pdelements[
            (self.pdelements["is_switch"] == True)
            & (self.pdelements["is_open"] == False)
        ].reset_index(drop=True)

        # normally open tie switches
        if self.DERs is not None:
            # if DERs exists, tie switches should be differentiated from virtual switches
            # tie switches
            tie_switch_names = self.normally_open_components[
                ~self.normally_open_components["normally_open_components"].isin(
                    self.DERs["name"]
                )
            ]
            self.model.tie_switches = self.pdelements[
                self.pdelements["name"].isin(
                    tie_switch_names["normally_open_components"]
                )
            ].reset_index(drop=True)

            # virtual switches
            self.model.virtual_switches = self.pdelements[
                self.pdelements["name"].isin(self.DERs["name"])
            ].reset_index(drop=True)

        else:
            self.model.tie_switches = self.pdelements[
                self.pdelements["name"].isin(
                    self.normally_open_components["normally_open_components"]
                )
            ].reset_index(drop=True)

            # virtual switches will be None if DERs do not exist
            self.model.virtual_switches = None

        # access DERs and extract info for virtual switches
        self.model.DERs = self.DERs

        # list of nodes and edges in the network
        self.model.nodes = list(self.network_tree.nodes())
        self.model.edges = list(self.network_tree.edges())

        # store the list of tuples for normally open components for future usage
        self.normally_open_tuples = []

        # we now add DERs and normally open switches to identify potential cycles in the network in any reconfiguration setup
        # keep graph separate from trees in a sense that graph can have cycles as we introduce it here by adding tie and virtual switches
        for _, row in self.pdelements[
            self.pdelements["name"].isin(
                self.normally_open_components["normally_open_components"]
            )
        ].iterrows():
            # update the edge and graph with additional edges i.e. tie switches and virtual switches -> if any
            self.model.edges.append((row["from_bus"], row["to_bus"]))
            self.network_graph.add_edge(
                row["from_bus"],
                row["to_bus"],
                # the remaining arguments are the data associated with each edges
                # nothing fancy just keeping this for additional info for eg. plot network
                element=row["element"],
                is_switch=row["is_switch"],
                is_open=row["is_open"],
                name=row["name"],
            )

            self.normally_open_tuples.append((row["from_bus"], row["to_bus"]))

        # obtain cycles in a network and convert the nodes to edge indices
        # self.model.cycles = network_cycles_simple(self.network_graph)
        self.model.cycles = network_cycles_basis(self.network_graph)
        self.model.loop_edge_idxs = loop_edges_to_tree_index(
            self.model.cycles, self.model.edges
        )
        logger.info(
            "Obtained cycles in the network and converted them to edge indices."
        )

        # from nodes and to nodes
        self.model.source_nodes, self.model.target_nodes = zip(*self.model.edges)

        # overall demand of the system (per phase)
        self.model.demand = self.loads
        self.model.active_demand_each_node = (
            self.model.demand["P1"] + self.model.demand["P2"] + self.model.demand["P3"]
        )
        self.model.total_demand = self.model.active_demand_each_node.sum()

        # number of nodes and edges from graph -> since they have tie and virtual switches added
        self.model.num_nodes = self.network_graph.number_of_nodes()
        self.model.num_edges = self.network_graph.number_of_edges()

        # dict that maps each line to their upstream and downstream switches
        self.model.line_to_switch_dict = self.lines_to_switch_mapper

        # switch indices
        self.model.sectionalizing_switch_indices = self.sectionalizing_switch_indices
        self.model.tie_switch_indices = self.tie_switch_indices
        self.model.virtual_switch_indices = self.virtual_switch_indices

        # hash map of node and edge indices in the tree
        self.model.node_indices_in_tree = self.node_indices_in_tree
        self.model.edge_indices_in_tree = self.edge_indices_in_tree

        # also store the faults here
        self.model.faults = self.faults

    @timethis
    def initialize_base_variables(self) -> None:
        """Initialize necessary variables for the optimization problem."""
        self.p_max = self.model.total_demand * BIG_M_POWER_MULTIPLIER
        self.p_min = -self.model.total_demand * BIG_M_POWER_MULTIPLIER

        self.model.v_i = RangeSet(0, self.model.num_nodes - 1)
        self.model.x_ij = RangeSet(0, self.model.num_edges - 1)

        # we do not need to care about DERs since the flow on their edge i.e. virtual edge will be covered by pdelements flow -> DER edges
        # connectivity
        self.model.vi = Var(self.model.v_i, bounds=(0, 1), domain=Binary)
        self.model.si = Var(self.model.v_i, bounds=(0, 1), domain=Binary)
        self.model.xij = Var(self.model.x_ij, bounds=(0, 1), domain=Binary)

        # power
        self.model.Pija = Var(
            self.model.x_ij, bounds=(self.p_min, self.p_max), domain=Reals
        )
        self.model.Pijb = Var(
            self.model.x_ij, bounds=(self.p_min, self.p_max), domain=Reals
        )
        self.model.Pijc = Var(
            self.model.x_ij, bounds=(self.p_min, self.p_max), domain=Reals
        )
        self.model.Qija = Var(
            self.model.x_ij, bounds=(self.p_min, self.p_max), domain=Reals
        )
        self.model.Qijb = Var(
            self.model.x_ij, bounds=(self.p_min, self.p_max), domain=Reals
        )
        self.model.Qijc = Var(
            self.model.x_ij, bounds=(self.p_min, self.p_max), domain=Reals
        )

        # voltage
        self.model.Via = Var(self.model.v_i, domain=Reals)
        self.model.Vib = Var(self.model.v_i, domain=Reals)
        self.model.Vic = Var(self.model.v_i, domain=Reals)

    @timethis
    def initialize_user_variables(self) -> None:
        """initialize user defined variables"""
        logger.info("Currently unavailable.This will be updated in the near future ...")
        pass

    @timethis
    def initialize_user_constraints(self) -> None:
        """initialize user defined constraints"""
        logger.info("Currently unavailable.This will be updated in the near future ...")
        pass

    @timethis
    def constraints_base(
        self,
        base_kV_LL: float = None,
        vmax: float = 1.05,
        vmin: float = 0.95,
        vsub: float = 1.05,
        psub_max: float = 5000,
    ) -> None:
        """Contains all of the constraints in the base restoration model

        Args:
            base_kV_LL (float): basekV of the circuit.
            This should be technically different for each transformer sections. Wll update in the future version.
            vmax (float, optional): Maximum voltage of the system in per unit (pu). Defaults to 1.05.
            vmin (float, optional): Minimum voltage of the system in per unit (pu). Defaults to 0.95.
            vsub (float, optional): reference substation voltage of the system in per unit (pu). Defaults to 1.05.
            psub_max (float, optional): maximum substation flow of the system in kW. Defaults to 5000 kW.

        """
        if not base_kV_LL:
            logger.error(
                "base_kV_LL is a required argument. Please provide the line to line base kV for this circuit"
            )
            raise missing_kVbase_error(
                "base_kV_LL is missing. Please provide line to line base kV for this circuit."
            )

        self._constraints_list = []

        # ------------------------------- connectivity constraints ------------------------------------

        # ---------------------------------------------------------------------------------------------
        @timethis
        def connectivity_constraint_rule_base() -> None:
            """Constraints for network connectivity and node-edge energization."""

            def connectivity_si_rule(model, i) -> Constraint:
                return model.si[i] <= model.vi[i]

            def connectivity_vi_rule(model, i) -> Constraint:
                return (
                    model.xij[i]
                    <= model.vi[self.node_indices_in_tree[model.source_nodes[i]]]
                )

            def connectivity_vj_rule(model, i) -> Constraint:
                return (
                    model.xij[i]
                    <= model.vi[self.node_indices_in_tree[model.target_nodes[i]]]
                )

            self.model.connectivity_si = Constraint(
                self.model.v_i, rule=connectivity_si_rule
            )
            self.model.connectivity_vi = Constraint(
                self.model.x_ij, rule=connectivity_vi_rule
            )
            self.model.connectivity_vj = Constraint(
                self.model.x_ij, rule=connectivity_vj_rule
            )

            logger.info(
                f"Successfully added connectivity constraints as {self.model.connectivity_vi} and {self.model.connectivity_vj}"
            )

            # append these constraints for user information
            self._constraints_list.append(self.model.connectivity_si)
            self._constraints_list.append(self.model.connectivity_vi)
            self._constraints_list.append(self.model.connectivity_vj)

        # generate constraints for network connectivity
        connectivity_constraint_rule_base()

        # ------------------------------- branch flow constraints ------------------------------------

        # ---------------------------------------------------------------------------------------------
        @timethis
        def powerflow_rule_base() -> None:
            """Constraints for power flow balance in each of the nodes such that incoming power is the sum of outgoing power
            and nodal demand.
             - This is the base power flow and does not consider any backup resources (PV, battery, or other backup DGs)
            """
            self.model.power_flow = ConstraintList()

            for k in self.model.x_ij:
                active_node = self.model.target_nodes[k]
                active_node_index = self.node_indices_in_tree[active_node]

                children_nodes = [
                    ch_nodes
                    for ch_nodes, each_node in enumerate(self.model.source_nodes)
                    if each_node == active_node
                ]
                parent_nodes = [
                    pa_nodes
                    for pa_nodes, each_node in enumerate(self.model.target_nodes)
                    if each_node == active_node
                ]

                # access active and reactive power matching each indices
                (
                    active_power_A,
                    reactive_power_A,
                    active_power_B,
                    reactive_power_B,
                    active_power_C,
                    reactive_power_C,
                ) = self.model.demand[
                    self.model.demand["bus"] == self.model.target_nodes[k]
                ][
                    ["P1", "Q1", "P2", "Q2", "P3", "Q3"]
                ].values[
                    0
                ]

                # flow constraint: Pin = Pdemand (if picked up) + Pout (same for Q)
                self.model.power_flow.add(
                    sum(self.model.Pija[each_parent] for each_parent in parent_nodes)
                    - active_power_A * self.model.si[active_node_index]
                    == sum(self.model.Pija[each_child] for each_child in children_nodes)
                )
                self.model.power_flow.add(
                    sum(self.model.Qija[each_parent] for each_parent in parent_nodes)
                    - reactive_power_A * self.model.si[active_node_index]
                    == sum(self.model.Qija[each_child] for each_child in children_nodes)
                )

                # Phase B
                # flow constraint: Pin = Pdemand (if picked up) + Pout (same for Q)
                self.model.power_flow.add(
                    sum(self.model.Pijb[each_parent] for each_parent in parent_nodes)
                    - active_power_B * self.model.si[active_node_index]
                    == sum(self.model.Pijb[each_child] for each_child in children_nodes)
                )
                self.model.power_flow.add(
                    sum(self.model.Qijb[each_parent] for each_parent in parent_nodes)
                    - reactive_power_B * self.model.si[active_node_index]
                    == sum(self.model.Qijb[each_child] for each_child in children_nodes)
                )

                # Phase C
                # flow constraint: Pin = Pdemand (if picked up) + Pout (same for Q)
                self.model.power_flow.add(
                    sum(self.model.Pijc[each_parent] for each_parent in parent_nodes)
                    - active_power_C * self.model.si[active_node_index]
                    == sum(self.model.Pijc[each_child] for each_child in children_nodes)
                )
                self.model.power_flow.add(
                    sum(self.model.Qijc[each_parent] for each_parent in parent_nodes)
                    - reactive_power_C * self.model.si[active_node_index]
                    == sum(self.model.Qijc[each_child] for each_child in children_nodes)
                )
            logger.info(
                f"Successfully added power flow constraints as {self.model.power_flow}"
            )

            # append these constraints for user information
            self._constraints_list.append(self.model.power_flow)

        # generate constraints for power flow
        powerflow_rule_base()

        # ------------------------------- nodal voltage constraints ------------------------------------

        # ---------------------------------------------------------------------------------------------

        @timethis
        def voltage_limits_rule_base(vmax_sq: float, vmin_sq: float) -> None:
            """voltage limits in each of the nodes as per LinDistFlow

            Args:
            vmax_sq (float): Maximum square of the voltage of the system in per unit (pu).
            vmin_sq (float): Minimum square of the voltage of the system in per unit (pu).
            """

            self.model.voltage_limits = ConstraintList()

            for node in self.model.v_i:
                self.model.voltage_limits.add(self.model.Via[node] <= vmax_sq)
                self.model.voltage_limits.add(self.model.Vib[node] <= vmax_sq)
                self.model.voltage_limits.add(self.model.Vic[node] <= vmax_sq)
                self.model.voltage_limits.add(self.model.Via[node] >= vmin_sq)
                self.model.voltage_limits.add(self.model.Vib[node] >= vmin_sq)
                self.model.voltage_limits.add(self.model.Vic[node] >= vmin_sq)

            logger.info(
                f"Successfully added voltage limit constraints as {self.model.voltage_limits}"
            )

            # append these constraints for user information
            self._constraints_list.append(self.model.voltage_limits)

        # generate constraints for network connectivity
        voltage_limits_rule_base(vmax_sq=vmax**2, vmin_sq=vmin**2)

        @timethis
        def voltage_balance_rule_base() -> None:
            """voltage balance in each of the nodes as per LinDistFlow

            Reference: L. Gan and S. H. Low, "Convex relaxations and linear approximation for optimal power flow in multiphase radial networks,"
            2014 Power Systems Computation Conference, Wroclaw, Poland, 2014,

            """

            self.model.voltage_balance = ConstraintList()

            for _, each_line in self.model.pdelements.iterrows():
                edge_index = self.edge_indices_in_tree[
                    (each_line["from_bus"], each_line["to_bus"])
                ]

                # to remain consistent with the direction, we access nodes in the same order as the network edges
                # note: the order of network edges could be different than the pdelement edges (due to u,v and v,u convention)
                source_node_idx = self.node_indices_in_tree[each_line["from_bus"]]
                target_node_index = self.node_indices_in_tree[each_line["to_bus"]]

                # pandas save arrays as strings (or objects). So we evaluate them and reapply array type
                z_matrix_real = (
                    np.array(eval(each_line["z_matrix_real"])) * each_line["length"]
                )
                z_matrix_imag = (
                    np.array(eval(each_line["z_matrix_imag"])) * each_line["length"]
                )

                baseZ = each_line["base_kv_LL"] ** 2

                # ----------------------------------------------- Phase A -----------------------------------------------------------------
                if "a" in each_line["phases"]:

                    # obtain the resistance and reactance of the element
                    r_aa, x_aa, r_ab, x_ab, r_ac, x_ac = (
                        z_matrix_real[0, 0],
                        z_matrix_imag[0, 0],
                        z_matrix_real[0, 1],
                        z_matrix_imag[0, 1],
                        z_matrix_real[0, 2],
                        z_matrix_imag[0, 2],
                    )
                    if each_line["is_switch"]:
                        # model the voltage equations for lines with a switch
                        self.model.voltage_balance.add(
                            self.model.Via[source_node_idx]
                            - self.model.Via[target_node_index]
                            - 2
                            * r_aa
                            / (KW_TO_MW_FACTOR * baseZ)
                            * self.model.Pija[edge_index]
                            - 2
                            * x_aa
                            / (KW_TO_MW_FACTOR * baseZ)
                            * self.model.Qija[edge_index]
                            + (r_ab - np.sqrt(3) * x_ab)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Pijb[edge_index]
                            + (x_ab + np.sqrt(3) * r_ab)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Qijb[edge_index]
                            + (r_ac + np.sqrt(3) * x_ac)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Pijc[edge_index]
                            + (x_ac - np.sqrt(3) * r_ac)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Qijc[edge_index]
                            - BIG_M_VOLTAGE_MULTIPLIER
                            * (1 - self.model.xij[edge_index])
                            <= 0
                        )

                        self.model.voltage_balance.add(
                            self.model.Via[source_node_idx]
                            - self.model.Via[target_node_index]
                            - 2
                            * r_aa
                            / (KW_TO_MW_FACTOR * baseZ)
                            * self.model.Pija[edge_index]
                            - 2
                            * x_aa
                            / (KW_TO_MW_FACTOR * baseZ)
                            * self.model.Qija[edge_index]
                            + (r_ab - np.sqrt(3) * x_ab)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Pijb[edge_index]
                            + (x_ab + np.sqrt(3) * r_ab)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Qijb[edge_index]
                            + (r_ac + np.sqrt(3) * x_ac)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Pijc[edge_index]
                            + (x_ac - np.sqrt(3) * r_ac)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Qijc[edge_index]
                            + BIG_M_VOLTAGE_MULTIPLIER
                            * (1 - self.model.xij[edge_index])
                            >= 0
                        )
                    else:
                        # model the voltage equations for lines without switch
                        self.model.voltage_balance.add(
                            self.model.Via[source_node_idx]
                            - self.model.Via[target_node_index]
                            - 2
                            * r_aa
                            / (KW_TO_MW_FACTOR * baseZ)
                            * self.model.Pija[edge_index]
                            - 2
                            * x_aa
                            / (KW_TO_MW_FACTOR * baseZ)
                            * self.model.Qija[edge_index]
                            + (r_ab - np.sqrt(3) * x_ab)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Pijb[edge_index]
                            + (x_ab + np.sqrt(3) * r_ab)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Qijb[edge_index]
                            + (r_ac + np.sqrt(3) * x_ac)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Pijc[edge_index]
                            + (x_ac - np.sqrt(3) * r_ac)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Qijc[edge_index]
                            == 0
                        )

                        self.model.voltage_balance.add(
                            self.model.Via[source_node_idx]
                            - self.model.Via[target_node_index]
                            - 2
                            * r_aa
                            / (KW_TO_MW_FACTOR * baseZ)
                            * self.model.Pija[edge_index]
                            - 2
                            * x_aa
                            / (KW_TO_MW_FACTOR * baseZ)
                            * self.model.Qija[edge_index]
                            + (r_ab - np.sqrt(3) * x_ab)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Pijb[edge_index]
                            + (x_ab + np.sqrt(3) * r_ab)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Qijb[edge_index]
                            + (r_ac + np.sqrt(3) * x_ac)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Pijc[edge_index]
                            + (x_ac - np.sqrt(3) * r_ac)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Qijc[edge_index]
                            == 0
                        )

                # ----------------------------------------------- Phase B -----------------------------------------------------------------
                if "b" in each_line["phases"]:

                    # obtain the resistance and reactance of the element
                    r_ba, x_ba, r_bb, x_bb, r_bc, x_bc = (
                        z_matrix_real[1, 0],
                        z_matrix_imag[1, 0],
                        z_matrix_real[1, 1],
                        z_matrix_imag[1, 1],
                        z_matrix_real[1, 2],
                        z_matrix_imag[1, 2],
                    )
                    if each_line["is_switch"]:
                        # model the voltage equations for lines with a switch
                        self.model.voltage_balance.add(
                            self.model.Vib[source_node_idx]
                            - self.model.Vib[target_node_index]
                            - 2
                            * r_bb
                            / (KW_TO_MW_FACTOR * baseZ)
                            * self.model.Pijb[edge_index]
                            - 2
                            * x_bb
                            / (KW_TO_MW_FACTOR * baseZ)
                            * self.model.Qijb[edge_index]
                            + (r_ba + np.sqrt(3) * x_ba)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Pija[edge_index]
                            + (x_ba - np.sqrt(3) * r_ba)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Qija[edge_index]
                            + (r_bc - np.sqrt(3) * x_bc)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Pijc[edge_index]
                            + (x_bc + np.sqrt(3) * r_bc)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Qijc[edge_index]
                            - BIG_M_VOLTAGE_MULTIPLIER
                            * (1 - self.model.xij[edge_index])
                            <= 0
                        )

                        self.model.voltage_balance.add(
                            self.model.Vib[source_node_idx]
                            - self.model.Vib[target_node_index]
                            - 2
                            * r_bb
                            / (KW_TO_MW_FACTOR * baseZ)
                            * self.model.Pijb[edge_index]
                            - 2
                            * x_bb
                            / (KW_TO_MW_FACTOR * baseZ)
                            * self.model.Qijb[edge_index]
                            + (r_ba + np.sqrt(3) * x_ba)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Pija[edge_index]
                            + (x_ba - np.sqrt(3) * r_ba)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Qija[edge_index]
                            + (r_bc - np.sqrt(3) * x_bc)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Pijc[edge_index]
                            + (x_bc + np.sqrt(3) * r_bc)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Qijc[edge_index]
                            + BIG_M_VOLTAGE_MULTIPLIER
                            * (1 - self.model.xij[edge_index])
                            >= 0
                        )
                    else:
                        # model the voltage equations for lines without switch
                        self.model.voltage_balance.add(
                            self.model.Vib[source_node_idx]
                            - self.model.Vib[target_node_index]
                            - 2
                            * r_bb
                            / (KW_TO_MW_FACTOR * baseZ)
                            * self.model.Pijb[edge_index]
                            - 2
                            * x_bb
                            / (KW_TO_MW_FACTOR * baseZ)
                            * self.model.Qijb[edge_index]
                            + (r_ba + np.sqrt(3) * x_ba)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Pija[edge_index]
                            + (x_ba - np.sqrt(3) * r_ba)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Qija[edge_index]
                            + (r_bc - np.sqrt(3) * x_bc)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Pijc[edge_index]
                            + (x_bc + np.sqrt(3) * r_bc)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Qijc[edge_index]
                            == 0
                        )

                        self.model.voltage_balance.add(
                            self.model.Vib[source_node_idx]
                            - self.model.Vib[target_node_index]
                            - 2
                            * r_bb
                            / (KW_TO_MW_FACTOR * baseZ)
                            * self.model.Pijb[edge_index]
                            - 2
                            * x_bb
                            / (KW_TO_MW_FACTOR * baseZ)
                            * self.model.Qijb[edge_index]
                            + (r_ba + np.sqrt(3) * x_ba)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Pija[edge_index]
                            + (x_ba - np.sqrt(3) * r_ba)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Qija[edge_index]
                            + (r_bc - np.sqrt(3) * x_bc)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Pijc[edge_index]
                            + (x_bc + np.sqrt(3) * r_bc)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Qijc[edge_index]
                            == 0
                        )

                # ----------------------------------------------- Phase C -----------------------------------------------------------------
                if "c" in each_line["phases"]:

                    # obtain the resistance and reactance of the element
                    r_ca, x_ca, r_cb, x_cb, r_cc, x_cc = (
                        z_matrix_real[2, 0],
                        z_matrix_imag[2, 0],
                        z_matrix_real[2, 1],
                        z_matrix_imag[2, 1],
                        z_matrix_real[2, 2],
                        z_matrix_imag[2, 2],
                    )

                    if each_line["is_switch"]:
                        # model the voltage equations for lines with a switch
                        self.model.voltage_balance.add(
                            self.model.Vic[source_node_idx]
                            - self.model.Vic[target_node_index]
                            - 2
                            * r_cc
                            / (KW_TO_MW_FACTOR * baseZ)
                            * self.model.Pijc[edge_index]
                            - 2
                            * x_cc
                            / (KW_TO_MW_FACTOR * baseZ)
                            * self.model.Qijc[edge_index]
                            + (r_ca - np.sqrt(3) * x_ca)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Pija[edge_index]
                            + (x_ca + np.sqrt(3) * r_ca)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Qija[edge_index]
                            + (r_cb + np.sqrt(3) * x_cb)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Pijb[edge_index]
                            + (x_cb - np.sqrt(3) * r_cb)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Qijb[edge_index]
                            - BIG_M_VOLTAGE_MULTIPLIER
                            * (1 - self.model.xij[edge_index])
                            <= 0
                        )

                        self.model.voltage_balance.add(
                            self.model.Vic[source_node_idx]
                            - self.model.Vic[target_node_index]
                            - 2
                            * r_cc
                            / (KW_TO_MW_FACTOR * baseZ)
                            * self.model.Pijc[edge_index]
                            - 2
                            * x_cc
                            / (KW_TO_MW_FACTOR * baseZ)
                            * self.model.Qijc[edge_index]
                            + (r_ca - np.sqrt(3) * x_ca)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Pija[edge_index]
                            + (x_ca + np.sqrt(3) * r_ca)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Qija[edge_index]
                            + (r_cb + np.sqrt(3) * x_cb)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Pijb[edge_index]
                            + (x_cb - np.sqrt(3) * r_cb)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Qijb[edge_index]
                            + BIG_M_VOLTAGE_MULTIPLIER
                            * (1 - self.model.xij[edge_index])
                            >= 0
                        )
                    else:
                        # model the voltage equations for lines without switch
                        self.model.voltage_balance.add(
                            self.model.Vic[source_node_idx]
                            - self.model.Vic[target_node_index]
                            - 2
                            * r_cc
                            / (KW_TO_MW_FACTOR * baseZ)
                            * self.model.Pijc[edge_index]
                            - 2
                            * x_cc
                            / (KW_TO_MW_FACTOR * baseZ)
                            * self.model.Qijc[edge_index]
                            + (r_ca - np.sqrt(3) * x_ca)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Pija[edge_index]
                            + (x_ca + np.sqrt(3) * r_ca)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Qija[edge_index]
                            + (r_cb + np.sqrt(3) * x_cb)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Pijb[edge_index]
                            + (x_cb - np.sqrt(3) * r_cb)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Qijb[edge_index]
                            == 0
                        )

                        self.model.voltage_balance.add(
                            self.model.Vic[source_node_idx]
                            - self.model.Vic[target_node_index]
                            - 2
                            * r_cc
                            / (KW_TO_MW_FACTOR * baseZ)
                            * self.model.Pijc[edge_index]
                            - 2
                            * x_cc
                            / (KW_TO_MW_FACTOR * baseZ)
                            * self.model.Qijc[edge_index]
                            + (r_ca - np.sqrt(3) * x_ca)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Pija[edge_index]
                            + (x_ca + np.sqrt(3) * r_ca)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Qija[edge_index]
                            + (r_cb + np.sqrt(3) * x_cb)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Pijb[edge_index]
                            + (x_cb - np.sqrt(3) * r_cb)
                            / (baseZ * KW_TO_MW_FACTOR)
                            * self.model.Qijb[edge_index]
                            == 0
                        )

            logger.info(
                f"Successfully added voltage constraints as {self.model.voltage_balance}"
            )

            # append these constraints for user information
            self._constraints_list.append(self.model.voltage_balance)

        # generate constraints for voltage balance
        voltage_balance_rule_base()
        # ------------------------------- branch flow limits constraints ------------------------------------

        # ---------------------------------------------------------------------------------------------------

        @timethis
        def powerflow_limit_rule_base() -> None:
            """powerflow limit in each of the lines coupled by switching action
            -M * xij <= Pij <= M * xij
            """

            self.model.powerflow_limit = ConstraintList()
            for k in self.model.x_ij:
                self.model.powerflow_limit.add(
                    self.model.Pija[k] <= self.p_max * self.model.xij[k]
                )
                self.model.powerflow_limit.add(
                    self.model.Pijb[k] <= self.p_max * self.model.xij[k]
                )
                self.model.powerflow_limit.add(
                    self.model.Pijc[k] <= self.p_max * self.model.xij[k]
                )
                self.model.powerflow_limit.add(
                    self.model.Pija[k] >= -self.p_max * self.model.xij[k]
                )
                self.model.powerflow_limit.add(
                    self.model.Pijb[k] >= -self.p_max * self.model.xij[k]
                )
                self.model.powerflow_limit.add(
                    self.model.Pijc[k] >= -self.p_max * self.model.xij[k]
                )

                self.model.powerflow_limit.add(
                    self.model.Qija[k] <= self.p_max * self.model.xij[k]
                )
                self.model.powerflow_limit.add(
                    self.model.Qijb[k] <= self.p_max * self.model.xij[k]
                )
                self.model.powerflow_limit.add(
                    self.model.Qijc[k] <= self.p_max * self.model.xij[k]
                )
                self.model.powerflow_limit.add(
                    self.model.Qija[k] >= -self.p_max * self.model.xij[k]
                )
                self.model.powerflow_limit.add(
                    self.model.Qijb[k] >= -self.p_max * self.model.xij[k]
                )
                self.model.powerflow_limit.add(
                    self.model.Qijc[k] >= -self.p_max * self.model.xij[k]
                )

            logger.info(
                f"Successfully added power flow limit constraints as {self.model.powerflow_limit}"
            )

            # append these constraints for user information
            self._constraints_list.append(self.model.powerflow_limit)

        # generate constraints for flow limits
        powerflow_limit_rule_base()

        # ------------------------------- substation voltage constraints ------------------------------------

        # ------------------------------------------------------------------------------------------------

        @timethis
        def substation_voltage_rule_base() -> None:
            """ensure that substation voltage is at 1 per unit"""
            self.model.substation_voltage = ConstraintList()

            substation_index = self.node_indices_in_tree[
                self.circuit_data["substation"]
            ]

            self.model.substation_voltage.add(
                self.model.Via[substation_index] == vsub**2
            )
            self.model.substation_voltage.add(
                self.model.Vib[substation_index] == vsub**2
            )
            self.model.substation_voltage.add(
                self.model.Vic[substation_index] == vsub**2
            )

            logger.info(
                f"Successfully added substation voltage (or subtransmission) initialization constraint at 1 pu as {self.model.substation_voltage}"
            )

            # append these constraints for user information
            self._constraints_list.append(self.model.substation_voltage)

        # generate constraints for substation voltage
        substation_voltage_rule_base()

        # ------------------------------- substation positive flow constraints ------------------------------------

        # ------------------------------------------------------------------------------------------------

        @timethis
        def substation_positiveflow_rule_base() -> None:
            """ensure that substations will always have positive flow"""
            self.model.substation_positive_flow = ConstraintList()

            # self.edge_indices_in_tree[(each_line['from_bus'], each_line['to_bus'])]
            substation_index = self.node_indices_in_tree[
                self.circuit_data["substation"]
            ]

            self.model.substation_positive_flow.add(
                self.model.Pija[substation_index] >= 0
            )
            self.model.substation_positive_flow.add(
                self.model.Pijb[substation_index] >= 0
            )
            self.model.substation_positive_flow.add(
                self.model.Pijc[substation_index] >= 0
            )

            self.model.substation_positive_flow.add(
                self.model.Pija[substation_index] <= psub_max
            )
            self.model.substation_positive_flow.add(
                self.model.Pijb[substation_index] <= psub_max
            )
            self.model.substation_positive_flow.add(
                self.model.Pijc[substation_index] <= psub_max
            )

            logger.info(
                f"Successfully added substation positive flow constraint as {self.model.substation_positive_flow}"
            )

            # append these constraints for user information
            self._constraints_list.append(self.model.substation_positive_flow)

        # generate constraints for substation flow such that their is negative flow in the substation
        substation_positiveflow_rule_base()

        # ----------------------------------- radiality constraints -----------------------------------------

        # ---------------------------------------------------------------------------------------------------

        @timethis
        def radiality_rule_base() -> None:
            """constraint to ensure radial configurations across all cycles. i.e. |N| <= |E| - 1.
            Since the cycles are simple cycles for each cycles |N| = |E| so by ensuring |E| - 1 we maintain radiality.
            """
            self.model.radiality = ConstraintList()
            for loops in self.model.loop_edge_idxs:
                # to ensure only switches are toogled to maintain radiality we observe switches in each loop
                switches_in_loop = set(loops) & self.all_switch_indices
                self.model.radiality.add(
                    sum(self.model.xij[edge_index] for edge_index in switches_in_loop)
                    <= len(switches_in_loop) - 1
                )
            msg_radiality = (
                f"Successfully added radiality constraint as {self.model.radiality}"
            )
            logger.info(msg_radiality)

            # append these constraints for user information
            self._constraints_list.append(self.model.radiality)

        # generate constraints for radiality
        radiality_rule_base()

        # ----------------------------------- fault constraints----------------------------------------------

        # ---------------------------------------------------------------------------------------------------

        @timethis
        def fault_sectionalizer_rule_base() -> None:
            """constraint to ensure the lines with the faults are opened or sectionalized.
            For base network every line is assumed to be sectionalizer
            """
            self.model.fault_sectionalize = ConstraintList()

            for fault in self.model.faults:

                # if fault is in a sectionalizer then the sectionalizer opens itself
                try:
                    fault_in_switch = (
                        True
                        if self.edge_indices_in_tree[fault]
                        in self.sectionalizing_switch_indices
                        else False
                    )
                    if fault_in_switch:
                        fault_sectionalizers = [fault]
                except KeyError:
                    try:
                        fault_in_switch = (
                            True
                            if self.edge_indices_in_tree[fault[::-1]]
                            in self.sectionalizing_switch_indices
                            else False
                        )
                        if fault_in_switch:
                            fault_sectionalizers = [fault[::-1]]
                    except KeyError:
                        msg_non_existing_edge = (
                            f"The edge {fault} does not exist in the network."
                        )
                        logger.error(msg_non_existing_edge)
                        raise KeyError(
                            f"{fault} is either invalid or does not exist in the network. Please provide a valid edge with fault."
                        )

                # if fault is in a line then we identify the switches(sectionalizers + tie switches) that open in order to isolate the fault
                if not fault_in_switch:
                    try:
                        fault_sectionalizers = self.model.line_to_switch_dict[fault]
                    except KeyError:
                        try:
                            fault_sectionalizers = self.model.line_to_switch_dict[
                                fault[::-1]
                            ]
                        except KeyError:
                            logger.error(
                                f"The edge {fault} does not exist in the network."
                            )
                            raise KeyError(
                                f"{fault} is either invalid or does not exist in the network. Please provide a valid edge with fault."
                            )

                logger.info(
                    f"The fault at {fault} is isolated by {fault_sectionalizers}."
                )

                for each_sectionalizer in fault_sectionalizers:
                    try:
                        sectionalizer_index = self.edge_indices_in_tree[
                            tuple(each_sectionalizer)
                        ]
                    except KeyError:
                        # this additional check is for sanity check.
                        logger.error(
                            f"The edge {each_sectionalizer} does not exist in the network."
                        )
                        raise KeyError(
                            f"{each_sectionalizer} is either invalid or does not exist in the network. Please provide a valid sectionalizer."
                        )

                    self.model.fault_sectionalize.add(
                        self.model.xij[sectionalizer_index] == 0
                    )

            logger.info(
                f"Successfully added fault sectionalizing constraint as {self.model.fault_sectionalize}."
            )

            # append these constraints for user information
            self._constraints_list.append(self.model.fault_sectionalize)

        # generate constraints for fault sectionalizer
        fault_sectionalizer_rule_base()

        # ----------------------------------------- DER limits ----------------------------------------------

        # ---------------------------------------------------------------------------------------------------

        @timethis
        def der_limit_rule_base() -> None:
            """active power limits from DERs"""
            self.model.der_limits = ConstraintList()
            for _, each_row in self.model.DERs.iterrows():

                # access edge indices from the network
                edge_index = self.edge_indices_in_tree[
                    (self.circuit_data["substation"], each_row["connected_bus"])
                ]
                self.model.der_limits.add(
                    self.model.Pija[edge_index]
                    + self.model.Pijb[edge_index]
                    + self.model.Pijc[edge_index]
                    <= each_row["kW_rated"]
                )

                # lower limit of DERs should be 0
                # self.model.der_limits.add(self.model.Pija[edge_index] >= 0)
                # self.model.der_limits.add(self.model.Pijb[edge_index] >= 0)
                # self.model.der_limits.add(self.model.Pijc[edge_index] >= 0)

            logger.info(
                f"Successfully added DERs active power limit constraint as {self.model.der_limits}."
            )

            # append these constraints for user information
            self._constraints_list.append(self.model.der_limits)

        if self.model.DERs is not None:
            # generate constraints for DER limits if DERs are included
            der_limit_rule_base()

        logger.info(
            "The variables and constraints for the base restoration model has been successfully loaded."
        )

    @timethis
    def objective_load_only(self) -> None:
        """Objective to minimize the loss of load or maximize the total load pick up"""

        self.model.restoration_objective = Objective(
            expr=(
                sum(
                    self.model.si[i]
                    * self.model.active_demand_each_node[
                        self.demand_node_indices_in_tree[i]
                    ]
                    for i in self.model.v_i
                )
            ),
            sense=maximize,
        )

    @timethis
    def objective_load_and_switching(
        self, alpha: float = 1.0, beta: float = 0.2
    ) -> None:
        """Objective to minimize the loss of load or maximize the total load pick up

        Args:
            alpha (float, optional): Weight factor for load restoration (load pick up maximization) objective. Defaults to 1
            beta (float, optional): Weight factor for switching minimization. Defaults to 0.2
        """
        self.model.restoration_objective = Objective(
            expr=(
                alpha
                * sum(
                    self.model.si[i]
                    * self.model.active_demand_each_node[
                        self.demand_node_indices_in_tree[i]
                    ]
                    for i in self.model.v_i
                )
                + beta
                * sum(
                    self.model.xij[j] for j in self.model.sectionalizing_switch_indices
                )
                - beta * sum(self.model.xij[k] for k in self.model.tie_switch_indices)
            ),
            sense=maximize,
        )

    @timethis
    def objective_load_switching_and_der(
        self, alpha: float = 1.0, beta: float = 0.2, gamma: float = None
    ) -> None:
        """Objective to minimize the loss of load and switching actions (tie + virtual)

        Args:
            alpha (float, optional): Weight factor for load restoration (load pick up maximization) objective. Defaults to 1
            beta (float, optional): Weight factor for switching minimization. Defaults to 0.2
            gamma (float, optional): Weight factor for virtual switch minimization. Defaults to 2 * beta * nos. of tie switches
            as per following reference: https://ieeexplore.ieee.org/document/9163237
        """

        if not gamma:
            # if gamma is not specified then put it as 2 * beta * number of tie switches as per following reference
            gamma = 2 * len(self.model.tie_switch_indices) * beta

        if (self.model.virtual_switches is None) or (not self.virtual_switch_indices):
            logger.error(
                "Incorrect objective accessed. Please use objective without DERs"
            )
            raise NotImplementedError(
                f"Cannot use objective_load_and_switching() without DERs. Please either include DERs or use 'objective_load_only()' objective"
            )

        self.model.restoration_objective = Objective(
            expr=(
                alpha
                * sum(
                    self.model.si[i]
                    * self.model.active_demand_each_node[
                        self.demand_node_indices_in_tree[i]
                    ]
                    for i in self.model.v_i
                )
                + beta
                * sum(
                    self.model.xij[j] for j in self.model.sectionalizing_switch_indices
                )
                - beta * sum(self.model.xij[k] for k in self.model.tie_switch_indices)
                - gamma
                * sum(self.model.xij[l] for l in self.model.virtual_switch_indices)
            ),
            sense=maximize,
        )

    @timethis
    def solve_model(
        self,
        solver: str = "gurobi",
        write_lp: bool = False,
        lp_filename: str = None,
        save_results: bool = False,
        results_filename: str = None,
        io_options: dict = {"symbolic_solver_labels": True},
        solver_options: dict = None,
        **kwargs,
    ) -> ConcreteModel | Any:
        """Solve the optimization model.

        Args:
            solver (str, optional): Solver to use (gurobi, cplex, glpk, ...). Defaults to 'gurobi'.
            write_lp (bool, optional): Whether to write lp file of the problem or not. Defaults to False
            lp_filename(str, optional): Filename for the lpfile
            save_results (bool, optional): Whether to save the results or not (currently supports CSV only). Defaults to False
            results_filename: str = filename for saving the results,
            io_options (dict, optional): input output options eg. symbolic solver labels. Defaults to {'symbolic_solver_labels': True}
            solver_options (dict, optional): solver related parameters. for eg. {'FeasibilityTol': 1e-3}
            **kwargs (dict): additional options as available in pyomo (for write, SolverFactory, and solve)

            see more of these kwargs and other available options in Pyomo (https://pyomo.readthedocs.io/en/stable/index.html)
            or solver documentations
        Returns:
            model (ConcreteModel): Solved concrete model object
        """

        if solver_options is None:
            solver_options = {}

        if io_options is None:
            io_options = {}

        if write_lp:
            if not lp_filename:
                lp_filename = "check.lp"
            self.model.write(lp_filename, io_options=io_options, **kwargs)

        if solver == "gurobi":
            optimization_solver = SolverFactory(
                solver, solver_io="python", solver_options=solver_options, **kwargs
            )
        else:
            optimization_solver = SolverFactory(solver)

        results = optimization_solver.solve(self.model, **kwargs)

        logger.info("Solved the restoration model successfully.")

        if save_results:
            self.save_variable_results(results, results_filename=results_filename)

        return self.model, results

    @timethis
    def save_variable_results(
        self, results: SolverResults, results_filename: str = None
    ) -> None:
        """Save the pyomo results in a file. Currently only CSVs are supported.

        Args:
            results (SolverResults): Pyomo solver results
            results_filename (str, optional): Filename of the results file. Defaults to None.
        """

        if (
            results.solver.status == SolverStatus.ok
            and results.solver.termination_condition == TerminationCondition.infeasible
        ):
            logger.warning("The model is infeasible. No solution file generated")
            return None

        # if filename is not defined then provide a default filename
        if not results_filename:
            results_filename = "solution.csv"

        all_results = []
        for variable in self.model.component_objects(Var, active=True):
            for index in variable:
                var_results = {}
                var_results["variable"] = variable.name
                var_results["index"] = index
                var_results["value"] = value(variable[index])
                all_results.append(var_results)

        # save the results in csv format
        pd.DataFrame(all_results).to_csv(results_filename, index=False)

        logger.info(f"The overall solution is saved in the file {results_filename}.")
