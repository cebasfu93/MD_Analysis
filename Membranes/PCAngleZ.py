#Membrane must be centered!

#CALCULATES THE XYZ COORDINATES OF EACH LIPID HEAGROUP AND THE ANGLE BETWEEN THE THE PHOSPHATE AND CHOLINE GROUPS 
XTC = "POPC2-24_PRO1-2_FIX.xtc"
TPR = "POPC2-24_PRO1.tpr"
NAME = XTC[:-8]

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from Extras import *
from MDAnalysis import *
plt.rcParams["font.family"] = "Arial"

U = Universe(TPR, XTC)
sel = {
"P31" : U.select_atoms("name P31"),
}

props_PCDistr={
"up"        : False,
"down"      : True,
"ref"       : 'P31',  #Used to find the membranes midplane
"angle_atoms"    : ["P31", "N31"], #The vector is drawn from the first to the second. They must be in the residue named PC
"start_ps"  : 25000,
"stop_ps"   : 100000,
"dt"        : 20,
}

def calc_angle(vectors1, vectors2):
    punto = np.diag(np.dot(vectors1, vectors2.T))
    norma1 = np.linalg.norm(vectors1, axis=1)
    norma2 = np.linalg.norm(vectors2, axis=1)
    normas = np.multiply(norma1, norma2)
    cos = np.divide(punto, normas)
    angs = np.arccos(cos)
    return angs

def PCAngleZ(props):
    if props['up']:
        sgn = 1
    else:
        sgn = -1
    z_ax = np.array([0,0,sgn])

    times = []
    results = {}

    g_ref = sel[props['ref']]
    g_a1 = U.select_atoms("resname PC and name {}".format(props['angle_atoms'][0]))
    g_a2 = U.select_atoms("resname PC and name {}".format(props['angle_atoms'][1]))

    for ts in U.trajectory:
        if ts.time >= props['start_ps'] and ts.time <= props['stop_ps'] and ts.time%props['dt']==0:
            times.append(ts.time)
            print("Time -> {:.1f} ps".format(ts.time))
            ndx_mols = np.where(sgn*(g_ref.positions[:,2] - ts.dimensions[2]/2) > 0)[0] #determines that the headgroup is in the right leaflet
            vecs = g_a2.positions[ndx_mols] - g_a1.positions[ndx_mols]
            v_ref = np.repeat([z_ax], len(vecs), axis=0)
            angles = calc_angle(vecs, v_ref)

            results["T{:.1f}_pos".format(ts.time)] = g_ref.positions[ndx_mols]
            results["T{:.1f}_ang".format(ts.time)] = angles

        if ts.time >= props['stop_ps']:
                break

    return times, results


def write_pcanglesz(time, angs_dict, props):
    f = open(NAME + "_pcz.sfu", 'w')

    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))

    f.write("#The following coordinates are those of the reference group\n")
    f.write("#{:<9} {:<9} {:<9} {:<13}\n".format("X (A)", "Y (A)", "Z (A)", "angle (rad)"))

    for i in range(len(time)):
        f.write("#T -> {:<10.3f} ps\n".format(time[i]))
        clp = "T{:.1f}_pos".format(time[i])
        cla = "T{:.1f}_ang".format(time[i])
        for j in range(len(angs_dict[cla])):
            f.write("{:<9.3f} {:<9.3f} {:<9.3f} {:<13.3f}\n".format(angs_dict[clp][j,0], angs_dict[clp][j,1], angs_dict[clp][j,2], angs_dict[cla][j]))
    f.close()


times, res_dict = PCAngleZ(props_PCDistr)
write_pcanglesz(times, res_dict, props_PCDistr)
