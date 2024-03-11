import logging 

# to avoid pyomo loggers getting printed set its level to only ERRORS
logging.getLogger('pyomo').setLevel(logging.ERROR)
logging.getLogger('gurobipy').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)