# -*- coding: utf-8 -*-
"""
Created on: April 12, 2021
@author: Abodh Poudyal
"""
breakpoint()
import restoration as rs
from fault_isolation import switch_identifier

# list of fault locations -> single or multiple
# python starts from 0
fault_location = [63]

# considers those cases in which fault occurs
if not fault_location == []:

    # obtain list of switches for fault isolation
    op_sw_fi = switch_identifier(fault_location)

    # restoration function that returns the following
    loss_objective = rs.restoration(op_sw_fi)
else:
    loss_objective = 0

print("Overall objective = ", loss_objective)







