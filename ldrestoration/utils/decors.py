from time import time
import logging

from ldrestoration.utils.logger_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

# decorator to time any of your function
def timethis(func_to_time):
    def timingwrapper(*args, **kwargs):
        start = time()
        result = func_to_time(*args, **kwargs)
        logger.info('The function "{}" took {:.2f} s to run'.format(func_to_time.__name__, time() - start))
        return result
    return timingwrapper 