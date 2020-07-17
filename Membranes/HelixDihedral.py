"""
Calculates rolling angle of an alpha helix. The rolling angle is defined as a dihedral between three atoms and the Z direction (1, 0, 0)
"""
XTC = "NP61-POPC6-46_PRO1_FIX.xtc" 
TPR = "NP61-POPC6-46_PRO1.tpr"
NAME = XTC[:-8]

import math
import numpy as np
import matplotlib.pyplot as plt
from Extras import *
from MDAnalysis import *
from scipy.spatial.distance import cdist
plt.rcParams["font.family"] = "Times New Roman"

U = Universe(TPR, XTC)
DT = U.trajectory[0].dt
sel = {
"TRP_BB" : U.select_atoms("index 214 293 352 421 490 559"),
"THR8_BB" : U.select_atoms("index 206 275 344 413 482 551"),
"THR6_BB" : U.select_atoms("index 200 269 338 407 476 545")
}

props_dih = {
'ref1'      : "TRP_BB",
'ref2'      : "THR6_BB",
'ref3'      : "THR8_BB",
'start_ps'  : 864000,
'stop_ps'   : 1000000,
'dt'        : 10
}

def norm_vec(vec):
    norm = np.array([np.linalg.norm(vec, axis=1)]).T
    return vec/norm

def helix_dihedral(props):
    g_ref1 = sel[props['ref1']]
    g_ref2 = sel[props['ref2']]
    g_ref3 = sel[props['ref3']]
    dih = []
    times = []
    for ts in U.trajectory:
        if ts.time >= props['start_ps'] and ts.time%props['dt']==0:
            d_temp = cdist(g_ref1.positions, g_ref2.positions)
            if np.all(d_temp < 90):
                #print(ts.time)
                x_ref1 = g_ref1.positions
                x_ref2 = g_ref2.positions
                x_ref3 = g_ref3.positions
                b1 = np.zeros((len(x_ref1), 3))
                b1[:,2] = 1
                b2 = norm_vec(x_ref2 - x_ref1)
                b3 = norm_vec(x_ref3 - x_ref2)
                n1 = norm_vec(np.cross(b1, b2))
                n2 = norm_vec(np.cross(b2, b3))
                m1 = norm_vec(np.cross(n1, b2))
                x_val = np.array([np.dot(i,j) for i,j in zip(n1, n2)])
                y_val = np.array([np.dot(i,j) for i,j in zip(m1, n2)])
                dih_tmp = np.arctan2(y_val, x_val)
                dih.append(dih_tmp)
                times.append(ts.time)
        if ts.time> props['stop_ps']:
            break
    dih = np.array(dih)*180/np.pi
    times = np.array(times)
    return times, dih

def write_dih(times, dihedrals, props):
    f = open(NAME+"_heldih.sfu", "w")
    f.write("#Angle of the alpha helix respect to its axis (deg)\n")
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    f.write("#time (ps)   Dihedral for every peptide (deg)\n")

    for t, dihs in zip(times, dihedrals):
        #print(dihs)
        f.write("{:<9.2f}".format(t) + (" {:>9.3f}"*len(dihs)).format(*dihs) + "\n")
    f.close()

times, dih = helix_dihedral(props_dih)
write_dih(times, dih, props_dih)
