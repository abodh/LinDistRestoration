"""
Created on: May 11, 2021
@author: Abodh Poudyal
"""

import numpy as np
from pyomo.environ import *
import Zmatrixa as zma
import Zmatrixb as zmb
import Zmatrixc as zmc


def restoration(F):
    # load data from text file
    edges = np.loadtxt('data/edges.txt')
    demand = np.loadtxt('data/load_data.txt')
    loops = np.loadtxt('data/cycles.txt')
    LineData = np.loadtxt('data/line_config.txt')

    # initialization
    nEdges = edges.shape[0]
    nNodes = demand.shape[0]
    fr = edges[:, 0]  # from node
    to = edges[:, 1]  # to node
    v_i = range(0, nNodes)  # bus energization variable range
    s_i = range(0, nNodes)  # load pick-up variable range
    x_ij = range(0, nEdges)  # line switching variable range
    V_i = range(0, nNodes)  # three-phase voltage vector

    Vmin = 0.95 ** 2
    Vmax = 1.05 ** 2

    # indices for virtual switches aka DG edges
    DG_edges = [126, 127, 128]

    # creating pyomo model
    model = ConcreteModel()

    # declaring pyomo variables
    model.vi = Var(v_i, bounds=(0, 1), within=Binary)
    model.si = Var(s_i, bounds=(0, 1), within=Binary)
    model.xij = Var(x_ij, bounds=(0, 1), within=Binary)
    model.xij0 = Var(x_ij, bounds=(0, 1), within=Binary)
    model.xij1 = Var(x_ij, bounds=(0, 1), within=Binary)
    model.Pija = Var(x_ij, bounds=(-10000, 10000), within=Reals)
    model.Pijb = Var(x_ij, bounds=(-10000, 10000), within=Reals)
    model.Pijc = Var(x_ij, bounds=(-10000, 10000), within=Reals)
    model.Qija = Var(x_ij, bounds=(-10000, 10000), within=Reals)
    model.Qijb = Var(x_ij, bounds=(-10000, 10000), within=Reals)
    model.Qijc = Var(x_ij, bounds=(-10000, 10000), within=Reals)
    model.Via = Var(V_i, bounds=(Vmin, Vmax), within=Reals)
    model.Vib = Var(V_i, bounds=(Vmin, Vmax), within=Reals)
    model.Vic = Var(V_i, bounds=(Vmin, Vmax), within=Reals)

    # load multiplier
    model.mult = (demand[:, 1] + demand[:, 3] + demand[:, 5])

    # critical load definition
    model.CL = np.ones(nNodes)

    # uncomment below if you want to add critical loads to the system
    # crit_load = np.array([[1, 3, 4, 5, 6, 52, 53, 54, 55, 56, 76, 77, 78, 79, 86, 98, 99, 100, 62, 63, 64, 65, 66,
    #                        47, 48, 49, 50, 28, 29, 30, 10, 11, 14]])
    # model.CL[crit_load - 1] = 10

    # pyomo objective
    model.value = Objective(expr=(sum(model.si[k] * model.mult[k] * model.CL[k] for k in s_i)), sense=maximize)

    # pyomo constraints
    model.c = ConstraintList()  # creates a list of constraints as placeholders

    # a. Constraints (s_i <= v_i)
    for k in range(0, nNodes):
        model.c.add(model.si[k] <= model.vi[k])

    # b. Constraints (x_ij <= v_i * v_j). This is non-linear and is linearized here.
    for k in range(0, nEdges):
        # here edges[k, 0] - 1] is used since our nodes name from 1 to 125 but python understands this as 0 to 124
        model.c.add(model.xij[k] <= model.vi[edges[k, 0] - 1])
        model.c.add(model.xij[k] <= model.vi[edges[k, 1] - 1])

    # c. Constraints (x_ij0 + x_ij1 <= x_ij)
    # prevents the forward and backward flow at the same time
    for k in range(0, nEdges):
        model.c.add(model.xij0[k] + model.xij1[k] <= model.xij[k])

    # d. Real power flow constraint for each of the Phases
    # Phase A
    for k in range(0, nEdges):
        ed = int(edges[k, 1] - 1)
        node = edges[k, 1]

        # Finding the all parent nodes of a particular node
        pa = np.array(np.where(to == edges[k, 1]))
        pa = pa.flatten()
        N = range(0, pa.__len__())

        # Finding the all children nodes of a particular node
        ch = np.array(np.where(fr == edges[k, 1]))
        ch = ch.flatten()
        M = range(0, ch.__len__())

        # overall equation
        # demand[ed, 1] = 0
        model.c.add(sum(model.Pija[pa[j]] for j in N) - demand[ed, 1] * model.si[node - 1] == \
                    sum(model.Pija[ch[j]] for j in M))

    # Phase B
    for k in range(0, nEdges):
        ed = int(edges[k, 1] - 1)
        node = edges[k, 1]

        # Finding the all parent nodes of a particular node
        pa = np.array(np.where(to == edges[k, 1]))
        pa = pa.flatten()
        N = range(0, pa.__len__())

        # Finding the all children nodes of a particular node
        ch = np.array(np.where(fr == edges[k, 1]))
        ch = ch.flatten()
        M = range(0, ch.__len__())

        # overall equation
        model.c.add(sum(model.Pijb[pa[j]] for j in N) - demand[ed, 3] * model.si[node - 1] == \
                    sum(model.Pijb[ch[j]] for j in M))

    # Phase C
    for k in range(0, nEdges):
        ed = int(edges[k, 1] - 1)
        node = edges[k, 1]

        # Finding the all parent nodes of a particular node
        pa = np.array(np.where(to == edges[k, 1]))
        pa = pa.flatten()
        N = range(0, pa.__len__())

        # Finding the all children nodes of a particular node
        ch = np.array(np.where(fr == edges[k, 1]))
        ch = ch.flatten()
        M = range(0, ch.__len__())

        # overall equation
        model.c.add(sum(model.Pijc[pa[j]] for j in N) - demand[ed, 5] * model.si[node - 1] == \
                    sum(model.Pijc[ch[j]] for j in M))

    # Now imposing the big-M method to ensure the real-power flowing in open line is zero
    # -M * x_ij0 <= Pij <= x_ij1* M
    M = 5000
    for k in range(0, nEdges):
        model.c.add(model.Pija[k] <= M * model.xij1[k])
        model.c.add(model.Pijb[k] <= M * model.xij1[k])
        model.c.add(model.Pijc[k] <= M * model.xij1[k])
        model.c.add(model.Pija[k] >= -M * model.xij0[k])
        model.c.add(model.Pijb[k] >= -M * model.xij0[k])
        model.c.add(model.Pijc[k] >= -M * model.xij0[k])

    # e. Reactive power flow constraint for each of the Phases
    # Phase A
    for k in range(0, nEdges):
        ed = int(edges[k, 1] - 1)
        node = edges[k, 1]

        # Finding the all parent nodes of a particular node
        pa = np.array(np.where(to == edges[k, 1]))
        pa = pa.flatten()
        N = range(0, pa.__len__())

        # Finding the all children nodes of a particular node
        ch = np.array(np.where(fr == edges[k, 1]))
        ch = ch.flatten()
        M = range(0, ch.__len__())

        # overall equation
        model.c.add(sum(model.Qija[pa[j]] for j in N) - demand[ed, 2] * model.si[node - 1] == \
                    sum(model.Qija[ch[j]] for j in M))

    # Phase B
    for k in range(0, nEdges):
        ed = int(edges[k, 1] - 1)
        node = edges[k, 1]

        # Finding the all parent nodes of a particular node
        pa = np.array(np.where(to == edges[k, 1]))
        pa = pa.flatten()
        N = range(0, pa.__len__())

        # Finding the all children nodes of a particular node
        ch = np.array(np.where(fr == edges[k, 1]))
        ch = ch.flatten()
        M = range(0, ch.__len__())

        # overall equation
        model.c.add(sum(model.Qijb[pa[j]] for j in N) - demand[ed, 4] * model.si[node - 1] == \
                    sum(model.Qijb[ch[j]] for j in M))

    # Phase C
    for k in range(0, nEdges):
        ed = int(edges[k, 1] - 1)
        node = edges[k, 1]

        # Finding the all parent nodes of a particular node
        pa = np.array(np.where(to == edges[k, 1]))
        pa = pa.flatten()
        N = range(0, pa.__len__())

        # Finding the all children nodes of a particular node
        ch = np.array(np.where(fr == edges[k, 1]))
        ch = ch.flatten()
        M = range(0, ch.__len__())

        # overall equation
        model.c.add(sum(model.Qijc[pa[j]] for j in N) - demand[ed, 6] * model.si[node - 1] == \
                    sum(model.Qijc[ch[j]] for j in M))

    # Now imposing the big-M method to ensure the reactive-power flowing in open line is zero
    # -M * x_ij0 <= Qij <= x_ij1* M
    M = 5000
    for k in range(0, nEdges):
        model.c.add(model.Qija[k] <= M * model.xij1[k])
        model.c.add(model.Qijb[k] <= M * model.xij1[k])
        model.c.add(model.Qijc[k] <= M * model.xij1[k])
        model.c.add(model.Qija[k] >= -M * model.xij0[k])
        model.c.add(model.Qijb[k] >= -M * model.xij0[k])
        model.c.add(model.Qijc[k] >= -M * model.xij0[k])

    # f. Voltage Constraints
    # base impedance
    '''
    calculation:
    base_Z = VLL in pu ** 2/ base_MVA

    for three phase, 
    -> base_Z = (VLL/sqrt(3)) ** 2 / (base_MVA/3)
    -> i.e., base_Z = VLL ** 2/ base MVA
    For eg: VLL = 4.16 kV and base MVA = 1 (i.e., basekVA = 1000 then, 

    base_Z = 4.16 ** 2
    '''

    base_Z = 4.16 ** 2  # base impedance
    bigM = 5

    # Phase A
    for k in range(0, nEdges):
        idx_config = np.where((LineData[:, 0:2] == edges[k]).all(axis=1))[0][0]
        conf = LineData[idx_config, 3]  # configuration of a line
        len = LineData[idx_config, 2]  # length of a line

        # Get the Z matrix for a line
        r_aa, x_aa, r_ab, x_ab, r_ac, x_ac = zma.Zmatrixa(conf)
        line = [edges[k, 0], edges[k, 1]]

        # pdb.set_trace()
        model.c.add(model.Via[int(line[0]) - 1] - model.Via[int(line[1]) - 1] - \
                    2 * r_aa * len / (5280 * base_Z * 1000) * model.Pija[k] - \
                    2 * x_aa * len / (5280 * base_Z * 1000) * model.Qija[k] + \
                    (r_ab - np.sqrt(3) * x_ab) * len / (5280 * base_Z * 1000) * model.Pijb[k] + \
                    (x_ab + np.sqrt(3) * r_ab) * len / (5280 * base_Z * 1000) * model.Qijb[k] + \
                    (r_ac + np.sqrt(3) * x_ac) * len / (5280 * base_Z * 1000) * model.Pijc[k] + \
                    (x_ac - np.sqrt(3) * r_ac) * len / (5280 * base_Z * 1000) * model.Qijc[k] - \
                    bigM * (1 - model.xij[k]) <= 0)

        model.c.add(model.Via[int(line[0]) - 1] - model.Via[int(line[1]) - 1] - \
                    2 * r_aa * len / (5280 * base_Z * 1000) * model.Pija[k] - \
                    2 * x_aa * len / (5280 * base_Z * 1000) * model.Qija[k] + \
                    (r_ab - np.sqrt(3) * x_ab) * len / (5280 * base_Z * 1000) * model.Pijb[k] + \
                    (x_ab + np.sqrt(3) * r_ab) * len / (5280 * base_Z * 1000) * model.Qijb[k] + \
                    (r_ac + np.sqrt(3) * x_ac) * len / (5280 * base_Z * 1000) * model.Pijc[k] + \
                    (x_ac - np.sqrt(3) * r_ac) * len / (5280 * base_Z * 1000) * model.Qijc[k] + \
                    bigM * (1 - model.xij[k]) >= 0)

    # Phase B
    for k in range(0, nEdges):
        idx_config = np.where((LineData[:, 0:2] == edges[k]).all(axis=1))[0][0]
        conf = LineData[idx_config, 3]  # configuration of a line
        len = LineData[idx_config, 2]  # length of a line

        # Get the Z matrix for a line
        r_bb, x_bb, r_ba, x_ba, r_bc, x_bc = zmb.Zmatrixb(conf)
        line = [edges[k, 0], edges[k, 1]]

        model.c.add(model.Vib[int(line[0]) - 1] - model.Vib[int(line[1]) - 1] - \
                    2 * r_bb * len / (5280 * base_Z * 1000) * model.Pijb[k] - \
                    2 * x_bb * len / (5280 * base_Z * 1000) * model.Qijb[k] + \
                    (r_ba + np.sqrt(3) * x_ba) * len / (5280 * base_Z * 1000) * model.Pija[k] + \
                    (x_ba - np.sqrt(3) * r_ba) * len / (5280 * base_Z * 1000) * model.Qija[k] + \
                    (r_bc - np.sqrt(3) * x_bc) * len / (5280 * base_Z * 1000) * model.Pijc[k] + \
                    (x_bc + np.sqrt(3) * r_bc) * len / (5280 * base_Z * 1000) * model.Qijc[k] - \
                    bigM * (1 - model.xij[k]) <= 0)

        model.c.add(model.Vib[int(line[0]) - 1] - model.Vib[int(line[1]) - 1] - \
                    2 * r_bb * len / (5280 * base_Z * 1000) * model.Pijb[k] - \
                    2 * x_bb * len / (5280 * base_Z * 1000) * model.Qijb[k] + \
                    (r_ba + np.sqrt(3) * x_ba) * len / (5280 * base_Z * 1000) * model.Pija[k] + \
                    (x_ba - np.sqrt(3) * r_ba) * len / (5280 * base_Z * 1000) * model.Qija[k] + \
                    (r_bc - np.sqrt(3) * x_bc) * len / (5280 * base_Z * 1000) * model.Pijc[k] + \
                    (x_bc + np.sqrt(3) * r_bc) * len / (5280 * base_Z * 1000) * model.Qijc[k] + \
                    bigM * (1 - model.xij[k]) >= 0)

    # Phase C
    for k in range(0, nEdges):
        idx_config = np.where((LineData[:, 0:2] == edges[k]).all(axis=1))[0][0]
        conf = LineData[idx_config, 3]  # configuration of a line
        len = LineData[idx_config, 2]  # length of a line

        # Get the Z matrix for a line
        r_cc, x_cc, r_ca, x_ca, r_cb, x_cb = zmc.Zmatrixc(conf)
        line = [edges[k, 0], edges[k, 1]]

        model.c.add(model.Vic[int(line[0]) - 1] - model.Vic[int(line[1]) - 1] - \
                    2 * r_cc * len / (5280 * base_Z * 1000) * model.Pijc[k] - \
                    2 * x_cc * len / (5280 * base_Z * 1000) * model.Qijc[k] + \
                    (r_ca - np.sqrt(3) * x_ca) * len / (5280 * base_Z * 1000) * model.Pija[k] + \
                    (x_ca + np.sqrt(3) * r_ca) * len / (5280 * base_Z * 1000) * model.Qija[k] + \
                    (r_cb + np.sqrt(3) * x_cb) * len / (5280 * base_Z * 1000) * model.Pijb[k] + \
                    (x_cb - np.sqrt(3) * r_cb) * len / (5280 * base_Z * 1000) * model.Qijb[k] - \
                    bigM * (1 - model.xij[k]) <= 0)

        model.c.add(model.Vic[int(line[0]) - 1] - model.Vic[int(line[1]) - 1] - \
                    2 * r_cc * len / (5280 * base_Z * 1000) * model.Pijc[k] - \
                    2 * x_cc * len / (5280 * base_Z * 1000) * model.Qijc[k] + \
                    (r_ca - np.sqrt(3) * x_ca) * len / (5280 * base_Z * 1000) * model.Pija[k] + \
                    (x_ca + np.sqrt(3) * r_ca) * len / (5280 * base_Z * 1000) * model.Qija[k] + \
                    (r_cb + np.sqrt(3) * x_cb) * len / (5280 * base_Z * 1000) * model.Pijb[k] + \
                    (x_cb - np.sqrt(3) * r_cb) * len / (5280 * base_Z * 1000) * model.Qijb[k] + \
                    bigM * (1 - model.xij[k]) >= 0)

    # reference voltage at the substation
    model.c.add(model.Via[124] == 1.048 ** 2)
    model.c.add(model.Vib[124] == 1.048 ** 2)
    model.c.add(model.Vic[124] == 1.048 ** 2)

    # g. Cyclic Constraints
    nC = loops.__len__()
    for k in range(0, nC):
        Sw = loops[k]
        nSw_C = np.count_nonzero(Sw)

        # radial feeder constraint
        model.c.add(sum(model.xij[Sw[j] - 1] for j in range(0, nSw_C)) <= nSw_C - 1)

    # h. Inserting fault in the System
    # isolate the line which has a fault or switch associated with it
    nF = F.__len__()
    for k in range(0, nF):
        model.c.add(model.xij[F[k]] == 0)

    # i. gen total limits
    # limit the power flow from DGs i.e., power flow on each virtual edge should be less or equal to DG size

    # model.c.add(sum(model.Pija[k] + model.Pijb[k] + model.Pijc[k] for k in DG_edges) <= 350)
    model.c.add(model.Pija[126] + model.Pijb[126] + model.Pijc[126] <= 120)  # 95
    model.c.add(model.Pija[127] + model.Pijb[127] + model.Pijc[127] <= 130)  # 122
    model.c.add(model.Pija[128] + model.Pijb[128] + model.Pijc[128] <= 100)  # 39

    # solver.set_instance(model)
    solver = SolverFactory('gurobi')
    results = solver.solve(model, tee=True)

    # value of the objective function which is the load loss after restoration
    obj = value(model.value)

    for k in range(0, nNodes):
        print("si[%d] = %f" % (k + 1, value(model.si[k])))

    for k in range(0, nEdges):
        print("xij[%d] = %f" % (k + 1, value(model.xij[k])))

    return obj