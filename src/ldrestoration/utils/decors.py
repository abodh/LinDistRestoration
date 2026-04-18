from time import time

from ldrestoration.utils.loggerconfig import logger


# decorator to time any of your function
def timethis(func_to_time):
    def timingwrapper(*args, **kwargs):
        start = time()
        result = func_to_time(*args, **kwargs)
        logger.debug(
            'The function "{}" took {:.2f} s to run'.format(func_to_time.__name__, time() - start)
        )
        return result

    return timingwrapper
