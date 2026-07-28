from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa

if TYPE_CHECKING:
    from altdss import AltDSS, Transformer
    from altdss.Transformer import TransformerBatch

from ldrestoration.dssparser.schema import TRANSFORMER_DATA
from ldrestoration.utils import logger


class TransformerHandler:
    def __init__(self, dss_instance: AltDSS) -> None:
        """Initialize a TransformerHandler instance. This instance deals with transformers in the distribution model.

        Args:
            dss_instance (AltDSS): redirected AltDSS instance
        """

        self.dss_instance: AltDSS = dss_instance

    def get_splitphase_primary(self) -> dict[str, str]:
        """Gets the primary phase information from split phase transformers to refer all loads to the primary

        Returns:
            splitphase_node_primary (dict[str,str]): A dictionary with secondary node as key and associated phase in primary as value
            for eg. for ['A.3', 'B.1.0', 'B.0.2'] this will return {'B':['3']}
        """
        msg: str = "We do not currently support delta connected primaries for split phase transformers."
        logger.warning(msg)
        breakpoint()

        splitphase_node_primary = {}
        transformer_flag = self.dss_instance.Transformers.First()
        while transformer_flag:
            if (self.dss_instance.CktElement.NumPhases() != 3) and self.dss_instance.Transformers.NumWindings() == 3:
                # a split phase transformer is a three winding single phase transformer (two phase primary accounts for delta)

                # name extracted of the secondary and phase extracted of the primary
                bus_name = self.dss_instance.CktElement.BusNames()[1].split(".")[0]
                bus_phases = self.dss_instance.CktElement.BusNames()[0].split(".")[1:]

                if bus_name not in splitphase_node_primary:
                    splitphase_node_primary[bus_name] = bus_phases

            transformer_flag = self.dss_instance.Transformers.Next()

        return splitphase_node_primary

    def _get_two_winding_transformer_data(self, two_winding_xfmrs: TransformerBatch) -> pa.Table:
        """Extract the two winding transformer data

        Args:
            two_winding_xfmrs (TransformerBatch): batch of two-winding transformers to extract the data from

        Returns:
            two_winding_data (pa.Table): Arrow table containing two-winding transformer data
        """

        name: np.ndarray = np.array(two_winding_xfmrs.Name)
        num_windings: np.ndarray = two_winding_xfmrs.Windings()
        num_phases: np.ndarray = two_winding_xfmrs.NumPhases()
        connected_from: np.ndarray = np.char.partition(two_winding_xfmrs.Buses, ".")[:, 0, 0]
        connected_to: np.ndarray = np.char.partition(two_winding_xfmrs.Buses, ".")[:, 1, 0]
        kva_rating: np.ndarray = np.array(two_winding_xfmrs.kVAs)[:, 0]
        primary_kv: np.ndarray = np.array(two_winding_xfmrs.kVs)[:, 0]
        secondary_kv: np.ndarray = np.array(two_winding_xfmrs.kVs)[:, 1]
        primary_connection: np.ndarray = np.array(two_winding_xfmrs.Conns_str)[:, 0]
        secondary_connection: np.ndarray = np.array(two_winding_xfmrs.Conns_str)[:, 1]
        tertiary_connection: np.ndarray = np.repeat("n/a", len(two_winding_xfmrs.Name))

        return pa.Table.from_arrays(
            [
                name,
                num_windings,
                num_phases,
                connected_from,
                connected_to,
                kva_rating,
                primary_kv,
                secondary_kv,
                primary_connection,
                secondary_connection,
                tertiary_connection,
            ],
            schema=TRANSFORMER_DATA,
        )

    def _get_three_winding_transformer_data(self, three_winding_xfmrs: TransformerBatch) -> pa.Table:
        """Extract the three winding transformer data

        Args:
            three_winding_xfmrs (TransformerBatch): batch of three-winding transformers to extract the data from

        Returns:
            three_winding_data (pa.Table): Arrow table containing three-winding transformer data
        """

        name: np.ndarray = np.array(three_winding_xfmrs.Name)
        num_windings: np.ndarray = three_winding_xfmrs.Windings()
        num_phases: np.ndarray = three_winding_xfmrs.NumPhases()
        connected_from: np.ndarray = np.char.partition(three_winding_xfmrs.Buses, ".")[:, 0, 0]
        connected_to: np.ndarray = np.char.partition(three_winding_xfmrs.Buses, ".")[:, 1, 0]
        kva_rating: np.ndarray = np.array(three_winding_xfmrs.kVAs)[:, 0]
        primary_kv: np.ndarray = np.array(three_winding_xfmrs.kVs)[:, 0]
        secondary_kv: np.ndarray = np.array(three_winding_xfmrs.kVs)[:, 1]
        primary_connection: np.ndarray = np.array(three_winding_xfmrs.Conns_str)[:, 0]
        secondary_connection: np.ndarray = np.array(three_winding_xfmrs.Conns_str)[:, 1]
        tertiary_connection: np.ndarray = np.array(three_winding_xfmrs.Conns_str)[:, 2]

        return pa.Table.from_arrays(
            [
                name,
                num_windings,
                num_phases,
                connected_from,
                connected_to,
                kva_rating,
                primary_kv,
                secondary_kv,
                primary_connection,
                secondary_connection,
                tertiary_connection,
            ],
            schema=TRANSFORMER_DATA,
        )

    def get_transformers(self) -> pa.Table:
        """Extract the transformer data from the distribution model.

        Returns:
            transformer_data (pa.Table): Arrow table containing transformer data for each transformer
        """
        transformer: Transformer = self.dss_instance.Transformer
        two_winding_xfmrs: TransformerBatch = transformer.batch(idx=np.where(np.array(transformer.Windings) == 2))
        three_winding_xfmrs: TransformerBatch = transformer.batch(idx=np.where(np.array(transformer.Windings) == 3))

        two_winding_data: pa.Table = self._get_two_winding_transformer_data(two_winding_xfmrs)
        three_winding_data: pa.Table = self._get_three_winding_transformer_data(three_winding_xfmrs)
        transformer_data: pa.Table = pa.concat_tables([two_winding_data, three_winding_data])

        msg: str = f"Extracted transformer data for {len(transformer_data)} transformers"
        logger.info(msg)
        return transformer_data
