import logging

from ldrestoration.utils.decors import timethis
from ldrestoration.utils.loggerconfig import setup_logging
setup_logging()
logger = logging.getLogger(__name__)


def line_unit_converter(current_unit: int) -> float:
    """Conversion factor of any existing units in OpenDSS to miles

    Args:
        current_unit (int): index of the unit in OpenDSS format
        from opendss manual -> length units index = {none|mi|kft|km|m|ft|in|cm}
    
    Returns:
        float: factor of conversion
    """    
    # 
    conversion = {
        0: 1,           # if none then we assume it is in miles
        1: 1,           # 1 mile = 1 mile
        2: 0.189,       # 1 kft = 0.18394 miles
        3: 0.621,       # 1 km = 0.621371 miles
        4: 6.21e-4,     # 1 m = 0.000621371 miles
        5: 1.89e-4,     # 1 ft = 0.000189394 miles
        6: 1.5783e-5,   # 1 inch = 1.5783e-5 miles
        7: 6.2137e-6    # 1 cm = 6.2137e-6 miles
    }
    
    #logger.debug(f"The conversion factor to convert {current_unit} to miles is {conversion[current_unit]}")
    return conversion[current_unit]