"""
Calculates the distance between the COM of the analytes and the COM of the metallic core
"""

XTC = "XXX_NVT_FIX.xtc"
TPR = "XXX_NVT.tpr"
NAME = XTC[:-8]

import math
import numpy as np
import matplotlib.pyplot as plt
from MDAnalysis import *
import MDAnalysis
plt.rcParams["font.family"] = "Arial"

U = Universe(TPR, XTC)
DT = U.trajectory[0].dt
sel = {
"all_gold" : U.select_atoms("resname AU AUS AUL"),
"analytes"       : U.select_atoms("resname A* and not resname AU AUS AUL"),
}

props_dist = {
'ref'       : "all_gold",
'target'    : "analytes",
'start_ps'  : 0,
'stop_ps'   : 10000,
}

def distance_to_com(props):
    print("Reference group: {}".format(props['ref']))
    g_ref = sel[props['ref']]
    print("Target group: {}".format(props['target']))
    g_target = sel[props['target']]
    times = []
    distances = []

    for ts in U.trajectory:
        if ts.time > props['stop_ps']:
            break
        if ts.time >= props['start_ps']:
            times.append(ts.time)
            ref_com = g_ref.center_of_mass()
            dist_frame = []
            for res in g_target.residues:
                dist = np.linalg.norm(res.atoms.center_of_mass() - ref_com)
                dist_frame.append(dist)
            distances.append(dist_frame)
    distances = np.array(distances)
    times = np.array(times)
    return times, distances

def write_distances(props, times, dists):
    n_anal = len(dists[0,:])
    f = open(NAME+"_dcom.sfu", "w")
    f.write("#Distance from target to the COM of ref in nm\n")
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    f.write("#time (ns) "+("R{:<7}"*n_anal).format(*np.linspace(1, n_anal, n_anal, dtype='int'))+"\n")
    for time, dist in zip(times/1000, dists/10):
        f.write("{:<10.3f} ".format(time))
        f.write(("{:<8.3f}"*n_anal).format(*dist))
        f.write("\n")

times, dists = distance_to_com(props_dist)
write_distances(props_dist, times, dists)
