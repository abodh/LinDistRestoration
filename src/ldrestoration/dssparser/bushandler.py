"Extract bus data from the distribution model via AltDSS API"

from typing import TYPE_CHECKING

import pyarrow as pa
import numpy as np

if TYPE_CHECKING:
    from altdss import AltDSS, Bus

from ldrestoration.dssparser.schema import BUS_DATA
from ldrestoration.utils import logger


class BusHandler:
    def __init__(self, dss_instance: AltDSS) -> None:
        """Initialize a BusHandler instance. This instance deals with bus (node) related data from the distribution model.
        Note: Bus and Nodes are two different concepts in distribution systems modeling and are used interchangably here
        for simplicity.

        Args:
            dss_instance (AltDSS): redirected AltDSS instance
        """

        self.dss_instance: AltDSS = dss_instance

    def get_buses(self) -> pa.Table:
        """Extract the bus data -> name, basekV, latitude, longitude from the distribution model.

        Returns:
            bus_data (pa.Table): Table containing bus data for each bus
        """

        bus: Bus = self.dss_instance.Bus
        name: np.ndarray = np.array(self.dss_instance.BusNames())
        kv_base: np.ndarray = np.array(bus.kVBase())
        latitude: np.ndarray = np.array(bus.Y())
        longitude: np.ndarray = np.array(bus.X())

        msg: str = f"Extracted bus data for {len(name)} buses."
        logger.info(msg)
        return pa.Table.from_arrays(
            [
                name,
                kv_base,
                latitude,
                longitude,
            ],
            schema=BUS_DATA,
        )
