#CALCULATES SECOND ORDER PARAMETER OF THE COATING THIOLS BASED ON THE C-C-C ANGLES
XTC = "NP18-53_PRO1_FIX.xtc" #Trajectory with NP centered
TPR = "NP18-53_PRO1.tpr"
NAME = XTC[:-8]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import mpl_toolkits.mplot3d.axes3d as axes3d
from scipy.spatial.distance import cdist
from Extras import *
from MDAnalysis import *
from physt import special
plt.rcParams["font.family"] = "Times New Roman"
z = 22

U = Universe(TPR, XTC)
DT = U.trajectory[0].dt
sel = {
"all_gold" : U.select_atoms("name AU AUS AUL"),
"L18" : U.select_atoms("resname L18")
}

props_order = {
'ref'       : "all_gold",
'ligand'    : "L18",
'anames'    : ["ST", "C1", "C2", "C4", "C5", "C6", "C7", "C8", "N1", "C9"],  #The atoms must be in the same order as going outwards from the core, and must belong to the same residue.
'start_ps'  : 25000,
'stop_ps'   : 100000,
'dt'        : 10,
}

def order(v_origin, v_target):
    cos = np.dot(v_origin, v_target)/(np.linalg.norm(v_origin)*np.linalg.norm(v_target))
    P2 = 0.5*(3*cos**2-1)
    return P2

def ligand_order(props):
    g_ref = sel[props['ref']]
    print("Reference COM: {}".format(props['ref']))
    g_anames = []
    for name in props['anames']:
        g_anames.append(U.select_atoms("name {}".format(name)))
    times = []
    results = {}

    for ts in U.trajectory:
        if ts.time >= props['start_ps'] and ts.time%props['dt']==0:
            times.append(ts.time)
            results["T{:.1f}_pos".format(ts.time)] = []
            results["T{:.1f}_ord".format(ts.time)] = []
            print("Time -> {:.1f} ps".format(ts.time))

            for res in sel[props['ligand']].residues:
                orders_res = []

                for a1, a2, a3 in zip(g_anames[:-2], g_anames[1:-1], g_anames[2:]):#, props['bonds_atoms'][:-1], props['bonds_atoms'][1:]):
                    g_a1 = res.atoms.intersection(a1)
                    g_a2 = res.atoms.intersection(a2)
                    g_a3 = res.atoms.intersection(a3)
                    v_ref = g_a2.positions[0] - g_ref.centroid()
                    v_bond =  g_a3.positions[0] - g_a1.positions[0]
                    p2 = order(v_ref, v_bond)
                    orders_res.append(p2)
                s_pos = res.atoms.intersection(g_anames[0]).positions[0] - g_ref.centroid()
                results["T{:.1f}_pos".format(ts.time)].append(s_pos)
                results["T{:.1f}_ord".format(ts.time)].append(orders_res)
            results["T{:.1f}_pos".format(ts.time)] = np.array(results["T{:.1f}_pos".format(ts.time)])
            results["T{:.1f}_ord".format(ts.time)] = np.array(results["T{:.1f}_ord".format(ts.time)])

        if ts.time >= props['stop_ps']:
                break
    return times, results

def write_porders(time, order_dict, props):
    f = open(NAME + "_order.sfu", 'w')

    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))

    f.write("#The following coordinates are with respect to the reference group's centroid\n")
    f.write("#{:<9} {:<9} {:<9} ".format("X (A)", "Y (A)", "Z (A)"))
    for name in props['anames'][1:-1]:
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

t, res = ligand_order(props_order)
write_porders(t, res, props_order)
