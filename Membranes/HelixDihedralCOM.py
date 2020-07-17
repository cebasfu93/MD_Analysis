"""
Calculates the rolling angle of an alpha-helix. The rolling angle is a dihedral between three atoms and the Z direction (0, 0, 1).
The rolling angle is reported as a function of the distance between COMs of the helix and the membrane.
"""
XTC = "MD/NP61-POPC6-46-r1_MD@@@.xtc"
TPR = "MD/NP61-POPC6-46-r1_MD@@@.tpr"
NAME = XTC[3:-4]

import math
import numpy as np
import matplotlib.pyplot as plt
from MDAnalysis import *
from scipy.spatial.distance import cdist
plt.rcParams["font.family"] = "Times New Roman"

U = Universe(TPR, XTC)
DT = U.trajectory[0].dt
sel = {
"TRP_BB" : U.select_atoms("index 214 293 352 421 490 559"),
"THR8_BB" : U.select_atoms("index 206 275 344 413 482 551"),
"THR6_BB" : U.select_atoms("index 200 269 338 407 476 545"),
"gH0" : U.select_atoms("resid 188:212"),
"gH1" : U.select_atoms("resid 213:237"),
"gH2" : U.select_atoms("resid 238:262"),
"gH3" : U.select_atoms("resid 263:287"),
"gH4" : U.select_atoms("resid 288:312"),
"gH5" : U.select_atoms("resid 313:337"),
"MEM" : U.select_atoms("resname POPC")
}

props_dih = {
'ref1'      : "TRP_BB",
'ref2'      : "THR6_BB",
'ref3'      : "THR8_BB",
'com_groups': ['gH0', 'gH1', 'gH2', 'gH3', 'gH4', 'gH5'],
'ref'       : 'MEM',
'start_ps'  : 0,
'stop_ps'   : 50000,
'dt'        : 10
}

def norm_vec(vec):
    norm = np.array([np.linalg.norm(vec, axis=1)]).T
    return vec/norm

def helix_dihedral(props):
    g_ref1 = sel[props['ref1']]
    g_ref2 = sel[props['ref2']]
    g_ref3 = sel[props['ref3']]
    g_coms = [sel[g_name] for g_name in props['com_groups']]
    g_ref = sel[props['ref']]
    dih = []
    coms = []
    for ts in U.trajectory:
        if ts.time >= props['start_ps'] and ts.time%props['dt']==0:
            d_temp = cdist(g_ref1.positions, g_ref2.positions)
            if np.all(d_temp < 90):
                coms_now = [np.abs(g_ref.center_of_mass()[2] - g.center_of_mass()[2]) for g in g_coms]
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
                coms.append(coms_now)
        if ts.time> props['stop_ps']:
            break
    dih = np.array(dih)*180/np.pi #deg to rad
    coms = np.array(coms)/10 #A to nm
    return coms, dih

def write_dih(coms, dihedrals, props):
    f = open("HELDIHCOM/"+NAME+"_heldihcom.sfu", "w")
    f.write("#Torsion of the alpha helix of peptides as a function of their individual Z distance to the COM of the membrane \n")
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    f.write("#COMS (nm)   Dihedrals (deg)\n")

    for com, dihs in zip(coms, dihedrals):
        f.write(("{:<9.2f} "*len(com)).format(*com) + (" {:>9.3f}"*len(dihs)).format(*dihs) + "\n")
    f.close()

coms, dih = helix_dihedral(props_dih)
write_dih(coms, dih, props_dih)
print("Done with {}".format(NAME))
