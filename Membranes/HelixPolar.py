"""
Calculates the polar angle of an alpha helix. The polar angle is defined between the helix's axis and the Z direction (1, 0, 0).
The polar angle is reported as a function of the distance between COMs of the helix and the membrane.
"""
XTC = "MD/gH-POPC5-46-r2_MD@@@.xtc"
TPR = "MD/gH-POPC5-46-r2_MD@@@.tpr"
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
"gH" : U.select_atoms("name BB SC1 SC2 SC3 SC4 SCN SCP"),
"MEM" : U.select_atoms("resname POPC"),
"LEU3_BB" : U.select_atoms("resid 3 and name BB"),
"ALA15_BB" : U.select_atoms("resid 15 and name BB"),
}

props_polar = {
'com_groups'  : ['gH', 'MEM'],
'ax_groups'   : ['LEU3_BB', 'ALA15_BB'],
'start_ps'  : 0,
'stop_ps'   : 50000,
'dt'        : 10,
}

def norm_vec(vec):
    return vec/np.linalg.norm(vec)

def angle_with_Z(vec):
    return np.arccos(np.clip(vec[2], -1, 1))

def helix_polar(props):
    g_com1, g_com2 = sel[props['com_groups'][0]], sel[props['com_groups'][1]]
    g_ax1, g_ax2 = sel[props['ax_groups'][0]], sel[props['ax_groups'][1]]

    comzs, angles = [], []

    for ts in U.trajectory:
        if ts.time >= props['start_ps'] and ts.time%props['dt']==0:
            comz = np.abs(g_com1.center_of_mass()[2] - g_com2.center_of_mass()[2])
            vec = norm_vec(g_ax2.positions[0] - g_ax1.positions[0])
            ang = angle_with_Z(vec)
            angles.append(ang)
            comzs.append(comz)
        if ts.time>props['stop_ps']:
            break
    angles = np.array(angles)*180/np.pi #rad to deg
    comzs = np.array(comzs)/10 # A to nm)
    return comzs, angles

def write_polar(comzs, angles, props):
    f = open("HELPOL/" + NAME+"_helpol.sfu", "w")
    f.write("#Angle between the N-C termini vector and the Z axis (deg) as a function of the distance between centers of mass in the Z direction (nm)\n")
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    f.write("#DZ (nm)   Polar angle (deg)\n")

    for com, ang in zip(comzs, angles):
        f.write("{:<10.3f} {:>10.3f}\n".format(com, ang))
    f.close()

comzs, angles = helix_polar(props_polar)
write_polar(comzs, angles, props_polar)
print("Done with {}".format(NAME))
