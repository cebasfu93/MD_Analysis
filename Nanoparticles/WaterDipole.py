#CALCULATES THE PROBABILITY DENSITY OF HAVING THE WATER DIPOLE MOMENTS A CERTAIN DISTANCE AWAY ALIGNED TO A CERTAIN ANGLE
XTC = "NP18-53_PRO1_FIX.xtc"
TPR = "NP18-53_PRO1.tpr"
NAME = XTC[:-8]

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist
from Extras import *
from MDAnalysis import *
plt.rcParams["font.family"] = "Times New Roman"

U = Universe(TPR, XTC)
DT = U.trajectory[0].dt
sel = {
"all_gold" : U.select_atoms("name AU AUS AUL"),
"OW"       : U.select_atoms("resname SOL and name OW"),
"HW"       : U.select_atoms("resname SOL and name HW1 HW2")
}

props_dipole = {
'ref'       : "all_gold",
'oxygen'    : "OW",
'hydrogen'  : "HW",
'start_ps'  : 25000,
'stop_ps'   : 100000,
'r_range'   : (10, 30),
'r_bins'    : 20,
'theta_bins': 90,
'dt'        : 40
}

def calculate_dipole(props):
    dipoles = np.zeros((props['theta_bins'], props['r_bins']))
    n_frames = ps_frame(props['stop_ps'], DT) - ps_frame(props['start_ps'], DT) + 1
    R = np.linspace(props['r_range'][0]/10, props['r_range'][1]/10, props['r_bins']+1)
    T = np.linspace(0, np.pi, props['theta_bins']+1)
    g_ref = sel[props['ref']]
    g_ow = sel[props['oxygen']]
    g_hw = sel[props['hydrogen']]
    print("Reference COM: {}".format(props['ref']))
    print("Oxygen group: {}".format(props['oxygen']))
    print("Hydrogen group: {}".format(props['hydrogen']))

    all_dists, all_angles = [], []
    for ts in U.trajectory:
        if ts.time >= props['start_ps'] and ts.time % props['dt'] == 0:
            print(ts.time)
            x_ref = g_ref.center_of_mass()
            x_ow = g_ow.positions
            x_ro = x_ow - x_ref
            dists = np.linalg.norm(x_ro, axis = 1)
            x_hw1 = g_hw.positions[::2]
            x_hw2 = g_hw.positions[1::2]
            x_hw = 0.5*(x_hw1 + x_hw2)
            x_ho = x_ow - x_hw
            angles = []
            for i in range(len(x_ow)):
                dot = np.dot(x_ho[i], x_ro[i])
                norms = np.linalg.norm(x_ho[i])*dists[i]
                angles.append(np.arccos(dot/norms))
            all_dists += list(dists)
            all_angles += angles
        if ts.time >= props['stop_ps']:
            break
    all_dists, all_angles = np.array(all_dists)/10, np.array(all_angles)
    pop, X, Y = np.histogram2d(all_dists, all_angles, bins=[R, T], normed = True)
    fig = plt.figure()
    plt.imshow(pop, interpolation='bilinear')
    plt.xticks([0, 22, 45], ['0', '90', '180'])
    plt.yticks([0, 5, 10], ['10', '20', '30'])
    plt.show()
    return X[:-1], Y[:-1], pop

def write_dipole(dists, angs, pops, props):
    f = open(NAME+"_dipole.sfu", "w")
    f.write("#Probability density density. 2D integral = 1\n")
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    f.write("#Distance (nm)/Angle (rad) ")
    for ang in angs:
        f.write("{:>9.3f} ".format(ang))
    f.write("\n")

    for i in range(len(dists)):
        f.write("{:<26.3f} ".format(dists[i]))
        for j in range(len(pops[0,:])):
            f.write("{:>9.5f} ".format(pops[i,j]))
        f.write("\n")
    f.close()

distances, angles, populations = calculate_dipole(props_dipole)
write_dipole(distances, angles, populations, props_dipole)
