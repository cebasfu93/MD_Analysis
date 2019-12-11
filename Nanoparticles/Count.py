#CALCULATES NUMBER OF MOLECULES (OF A GIVE GROUP) THAT FALLS WITHIN A CERTAIN RADIUS
XTC = "NP18-53_PRO1_FIX.xtc"
TPR = "NP18-53_PRO1.tpr"
NAME = XTC[:-8]

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from Extras import *
from MDAnalysis import *
plt.rcParams["font.family"] = "Times New Roman"

U = Universe(TPR, XTC)
DT = U.trajectory[0].dt
sel = {
"all_gold" : U.select_atoms("name AU AUS AUL"),
"L18" : U.select_atoms("resname L18"),
"NA"        : U.select_atoms("resname NA"),
"CL"        : U.select_atoms("resname CL"),
"SOL-OW"        : U.select_atoms("resname SOL and name OW"),
}

props_count = {
'ref'       : "L18",
'com'       : False, #If false, the reference will be all the atoms in the group ref
'targets'   : ["SOL-OW"],
'start_ps'  : 25000, #Only important for reported average and std. It will always calculate for all the trajectory
'stop_ps'   : 100000,
'd_max'     : 5, #Maximum distance to count (A)
'dt'        : 10
}

def count_species(props):
    counts = []
    g_ref = sel[props['ref']]
    print("Reference COM: {}".format(props['ref']))

    for target in props['targets']:
        times = []
        print("Current target: {}".format(target))
        g_target = sel[target]
        count = []
        for ts in U.trajectory:
            #if ts.time >= props['start_ps'] and ts.time%props['dt']==0:
            if ts.time%props['dt']==0:
                print(ts.time)
                if props['com']:
                    dists = np.linalg.norm(np.subtract(g_target.positions, g_ref.center_of_mass()), axis = 1)
                    c = np.sum(dists <= props['d_max'])
                else:
                    dists = cdist(g_target.positions, g_ref.positions)
                    c = np.sum(np.any(dists <= props['d_max'], axis = 1))
                times.append(ts.time)
                count.append(c)
            #if ts.time >= props['stop_ps']:
            #    break
        counts.append(count)
    counts = np.array(counts).T
    times = np.array(times)
    return times, counts

def write_count(times, counts, props):
    f = open(NAME+"_count.sfu", "w")
    f.write("#Counts (number)\n")
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    for k, key in enumerate(props['targets']):
        ndx_intime = np.logical_and(times >= props['start_ps'], times <= props['stop_ps'])
        av = np.mean(counts[ndx_intime,k])
        std = np.std(counts[ndx_intime, k])
        f.write("#Average for {}: {:.2f} +- {:.2f}\n".format(key, av, std))
    f.write("#time (ps)")
    for key in props['targets']:
        f.write("{:>9} ".format(key))
    f.write("\n")

    for i in range(len(times)):
        f.write("{:<9.2f} ".format(times[i]))
        for j in range(len(counts[0,:])):
            f.write("{:>9d} ".format(counts[i,j]))
        f.write("\n")
    f.close()

times, counts = count_species(props_count)
write_count(times, counts, props_count)
