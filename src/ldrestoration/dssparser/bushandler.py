import pandas as pd
import numpy as np
from altdss import AltDSS

from ldrestoration.utils import logger


class BusHandler:
    """BusHandler deals with bus (node) related data from the distribution model.

    Args:
        dss_instance (AltDSS): redirected opendssdirect instance

    Note:
        Bus and Nodes are two different concepts in distribution systems modeling and are used interchangably here
        for simplicity.
    """

    def __init__(self, dss_instance: AltDSS) -> None:
        """Initialize a BusHandler instance. This instance deals with bus (node) related data from the distribution model.
        Note: Bus and Nodes are two different concepts in distribution systems modeling and are used interchangably here
        for simplicity.

        Args:
            dss_instance (ModuleType): redirected opendssdirect instance
        """

        self.dss_instance = dss_instance

    def get_buses(self) -> pd.DataFrame:
        """Extract the bus data -> name, basekV, latitude, longitude from the distribution model.

        Returns:
            bus_data (pd.DataFrame): DataFrame containing bus data for each bus
        """

        bus = self.dss_instance.Bus
        bus_data = pd.DataFrame(
            {
                "name": np.array(self.dss_instance.BusNames()),
                "base_kv": bus.kVBase(),
                "latitude": bus.Y(),
                "longitude": bus.X(),
            }
        ).convert_dtypes(dtype_backend="pyarrow")
        msg = f"Extracted bus data for {len(bus_data)} buses."
        logger.info(msg)
        return bus_data
