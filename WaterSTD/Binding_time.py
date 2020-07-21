"""
For every binding event, it reports the minimum analyte-gold inter-COM distance and the binding residence time.
"""
XTC = "NP22sp-53_PRO1-10_FIX.xtc"
TPR = "NP22sp-53_PRO1.tpr"
NAME = XTC[:-8]

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from MDAnalysis import *

U = Universe(TPR, XTC)
DT = U.trajectory[0].dt
sel = {
"all_gold" : U.select_atoms("name AU AUS AUL"),
"mono_H"       : U.select_atoms("resname L22 and name H*"),
"SER_H"     : U.select_atoms("resname SER and name H*"),
"PHE_H"     : U.select_atoms("resname PHE and name H*"),
}

props_bind_time = {
'anchor'    : 'all_gold', #the script reports the minimum distance respect to this group
'ref'       : "mono_H",
'targets'    : ["SER_H", "PHE_H"],
'start_ps'  : 0,
'stop_ps'   : 1000000,
'd_max'     : 4, #A, threshold distance for magnetization transfer
}

flatten_list = lambda l: [item for sublist in l for item in sublist]

def bind_time(props):
    g_anchor = sel[props['anchor']]
    g_ref = sel[props['ref']]
    all_dists, all_bind_times = [], []
    for target in props['targets']:
        g_residues = [res.atoms for res in sel[target].residues]
        res_Hs = [res.intersection(sel[target]) for res in g_residues]
        bound = np.zeros((len(U.trajectory)+2, len(res_Hs)), dtype='int') #+2 to start and finish with False
        for t, ts in enumerate(U.trajectory,1):
            if ts.time % 100000 == 0:
                print(ts.time)
            if ts.time >= props['stop_ps']:
                break
            elif ts.time >= props['start_ps']:
                dists = [cdist(g_ref.positions, res.positions) for res in res_Hs]
                close = np.array([np.any(dist<= props['d_max']) for dist in dists])
                bound[t,close] = 1

        gates = bound[1:,:] - bound[:-1,:]
        on_switch, off_switch = [list(np.where(gate==1)[0]) for gate in gates.T], [list(np.where(gate==-1)[0]) for gate in gates.T]
        min_dists = [[np.min([np.linalg.norm(g_residues[i].center_of_mass() - g_anchor.center_of_mass()) for ts in U.trajectory[a:b]]) for a,b in zip(on, off)] for i, (on, off) in enumerate(zip(on_switch, off_switch))]
        durations = [[b-a for a,b in zip(on, off)] for on, off in zip(on_switch, off_switch)]
        min_dists = flatten_list(min_dists)
        durations = flatten_list(durations)

        all_dists.append(np.array(min_dists)/10) #A to nm
        all_bind_times.append(np.array(durations)*U.trajectory[0].dt) #frame to ps
    return all_dists, all_bind_times


def write_bind_time(props, dists, bind_times):
    f = open(NAME + "_btimes.sfu", 'w')
    values = []
    f.write("#Binding residence time (ps) from H-H contacts\n")
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))

    f.write("#Minimum inter-COM distance between anchor and target during the binding event (nm) \t Binding residence time (ps)\n")
    for target, dist, bind_time in zip(props['targets'], dists, bind_times):
        f.write("#TARGET GROUP: {}\n".format(target))
        for d, bt in zip(dist, bind_time):
            f.write("{:<10.3f}  {:>10.1f}\n".format(d, bt))
    f.close()

dists, bind_times = bind_time(props_bind_time)
write_bind_time(props_bind_time, dists, bind_times)
