#Membrane must be centered

#CALCULATES THE ANGLE FORMED BETWEEN THE PO4-NC3 VECTOR AND THE XY PROJECTION OF THE VECTOR BETWEEN THE CENTER OF MASS OF THE NP AND THE PO4 GROUPS
XTC = "NP18-POPC2-54_PRO1-37_FIX_D.xtc" #D has the membrane centered
TPR = "NP18-POPC2-54_PRO1.tpr"
NAME = XTC[:-10]

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from Extras import *
from MDAnalysis import *
plt.rcParams["font.family"] = "Arial"

U = Universe(TPR, XTC)
sel = {
"all_gold"  : U.select_atoms("name AU AUS AUL"),
"P31" : U.select_atoms("name P31"),
}

props_PCR={
"up"        : True,
"down"      : False,
"ref_np"    : 'all_gold',
"ref_mem"   : 'P31',  #Used to find the membranes midplane
"angle_atoms" : ["P31", "N31"], #The vector is drawn from the first to the second. They must be in the residue named PC
"start_ps"  : 200000,
"stop_ps"   : 500000,
"dt"        : 100,
}

def calc_angle(vectors1, vectors2):
    punto = np.diag(np.dot(vectors1, vectors2.T))
    norma1 = np.linalg.norm(vectors1, axis=1)
    norma2 = np.linalg.norm(vectors2, axis=1)
    normas = np.multiply(norma1, norma2)
    cos = np.divide(punto, normas)
    angs = np.arccos(cos)
    ndx_nan = np.isnan(angs)
    not_nan = np.invert(ndx_nan)
    ave = np.mean(angs[not_nan])
    angs[ndx_nan] = ave
    return angs

def PCAngleR(props):
    if props['up']:
        sgn = 1
    else:
        sgn = -1

    times = []
    results = {}

    g_mem = sel[props['ref_mem']]
    g_np = sel[props['ref_np']]
    g_a1 = U.select_atoms("resname PC and name {}".format(props['angle_atoms'][0]))
    g_a2 = U.select_atoms("resname PC and name {}".format(props['angle_atoms'][1]))

    for ts in U.trajectory:
        if ts.time >= props['start_ps'] and ts.time <= props['stop_ps'] and ts.time%props['dt']==0:
            times.append(ts.time)
            print("Time -> {:.1f} ps".format(ts.time))
            midplane = g_mem.center_of_mass()[2]
            ndx_mols = np.where(sgn*(g_mem.positions[:,2] - midplane) > 0)[0] #determines that the headgroup is in the right leaflet
            vecs12 = g_a2.positions[ndx_mols,0:-1] - g_a1.positions[ndx_mols,0:-1]
            v_np = g_np.center_of_mass()[0:-1]
            v_ref = g_a1.positions[ndx_mols,0:-1] - v_np
            angles = calc_angle(vecs12, v_ref)

            results["T{:.1f}_pos".format(ts.time)] = g_mem.positions[ndx_mols]
            results["T{:.1f}_ang".format(ts.time)] = angles

        if ts.time >= props['stop_ps']:
                break

    return times, results


def write_pcanglesr(time, angs_dict, props):
    f = open(NAME + "_pcr.sfu", 'w')

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


times, res_dict = PCAngleR(props_PCR)
write_pcanglesr(times, res_dict, props_PCR)
