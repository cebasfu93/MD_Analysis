#Membrane must be centered!

#CALCULATES THE ORDER PARAMETER BASED ON THE C-C-C ANGLES
XTC = "POPC2-24_PRO1-2_FIX.xtc"
TPR = "POPC2-24_PRO1.tpr"
NAME = XTC[:-8]

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from MDAnalysis import *
plt.rcParams["font.family"] = "Arial"
z = 22

U = Universe(TPR, XTC)
DT = U.trajectory[0].dt
sel = {
"P31" : U.select_atoms("name P31"),
"OL" : U.select_atoms("resname OL"),
"PA" : U.select_atoms("resname PA"),
}

props_order = {
'up'        : False,
'down'      : True,
'heads'     : 'P31',
'chains'    : ['OL', 'PA'], #residue names of the aliphatic chains
'anames'    : [["C21", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C110", "C111", "C112", "C113", "C114", "C115", "C116", "C117", "C118"],
                ["C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C110", "C111", "C112", "C113", "C114", "C115", "C116"]],  #Lists must be in order chain1-chain2. The atoms must be in the same order as going down the chain. The first atom of each list must be in PC
'start_ps'  : 0,
'stop_ps'   : 100000,
'dt'        : 100,
}

def calc_order(v_origin, v_target):
    cos = np.dot(v_origin, v_target)/(np.linalg.norm(v_origin)*np.linalg.norm(v_target))
    P2 = 0.5*(3*cos**2-1)
    return P2

def lipid_order(props):
    if props['up']:
        sgn = 1
    else:
        sgn = -1
    z_ax = np.array([0,0,sgn])

    times = []
    results = {}

    print("Reference group: {}".format(props['heads']))
    g_heads = sel[props['heads']]
    ndx_leaf = sgn*(g_heads.positions[:,2]-g_heads.center_of_mass()[2])>0
    g_leaf_heads = g_heads[ndx_leaf]
    g_anames = [[U.select_atoms("resname PC and name {}".format(props['anames'][0][0]))], [U.select_atoms("resname PC and name {}".format(props['anames'][1][0]))]]
    g_anames[0] += [U.select_atoms("resname {} and name {}".format(props['chains'][0], aname)) for aname in props['anames'][0][1:]]
    g_anames[1] += [U.select_atoms("resname {} and name {}".format(props['chains'][1], aname)) for aname in props['anames'][1][1:]]

    for ts in U.trajectory:
        if ts.time >= props['start_ps'] and ts.time <= props['stop_ps'] and ts.time%props['dt']==0:
            times.append(ts.time)
            results["T{:.1f}_pos".format(ts.time)] = []
            results["T{:.1f}_ord".format(ts.time)] = []
            print("Time -> {:.1f} ps".format(ts.time))
            for a, ah in enumerate(g_heads.positions):
                ord_res = []
                for g_tail in g_anames:
                    for g_a1, g_a2, g_a3 in zip(g_tail[2:], g_tail[1:-1], g_tail[:-2]):
                        #a1/2/3 have the same atoms of the same chains at a certain frame that are on the right leaflet
                        v31 = g_a1.positions[a] - g_a3.positions[a]
                        order = calc_order(v31, z_ax)
                        ord_res.append(order)
                results["T{:.1f}_pos".format(ts.time)].append(ah)
                results["T{:.1f}_ord".format(ts.time)].append(ord_res)
            results["T{:.1f}_pos".format(ts.time)] = np.array(results["T{:.1f}_pos".format(ts.time)])
            results["T{:.1f}_ord".format(ts.time)] = np.array(results["T{:.1f}_ord".format(ts.time)])

        if ts.time >= props['stop_ps']:
                break

    return times, results

def write_porders(time, order_dict, props):
    f = open(NAME + "_liporder.sfu", 'w')

    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))

    f.write("#The following coordinates are those of the heads\n")
    f.write("#{:<9} {:<9} {:<9} ".format("X (A)", "Y (A)", "Z (A)"))
    for c in range(len(props['chains'])):
        for name in props['anames'][c][1:-1]:
            f.write("{:<8}  ".format(name))
    f.write("\n")

    for i in range(len(time)):
        f.write("#T -> {:<10.3f} ps\n".format(time[i]))
        clp = "T{:.1f}_pos".format(time[i])
        clo = "T{:.1f}_ord".format(time[i])
        for j in range(len(order_dict[clo][:,0])):
            f.write("{:<9.3f} {:<9.3f} {:<9.3f} ".format(order_dict[clp][j,0], order_dict[clp][j,1], order_dict[clp][j,2]))
            for k in range(len(order_dict[clo][0,:])):
                f.write("{:<9.3f} ".format(order_dict[clo][j,k]))
            f.write("\n")
    f.close()

t, res = lipid_order(props_order)
write_porders(t, res, props_order)
