#CALCULATES THE PROBABILITY DENSITY (ALONG Z) OF FINDING ATOMS OF A CERTAIN GROUP

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
"N31" : U.select_atoms("name N31"),
}

props_PCDistr={
"ref"       : 'P31',  #Used to find the membranes midplane
"targets"    : ['P31', 'N31'],
"start_ps"  : 25000,
"stop_ps"   : 100000,
"dt"        : 10,
"z_range"   : (-35,35), #Angs from mem midplane
"z_bins"    : 140 #This will cover the whole height of the box
}

def PCDistribution(props):
    Z = np.linspace(props['z_range'][0], props['z_range'][1], props['z_bins']+1)
    ZC = center_bins(Z)
    g_ref = sel[props['ref']]
    print("Reference COM: {}".format(props['ref']))

    print("Targets: {}".format(props['targets']))
    distrs = {}
    zetas = {}
    g_targets = []
    for target in props['targets']:
        zetas[target] = []
        g_targets.append(sel[target])

    for ts in U.trajectory:
        if ts.time >= props['start_ps'] and ts.time <= props['stop_ps'] and ts.time % props['dt'] == 0:
            print(ts.time)
            midz = np.mean(g_ref.positions[:,2])
            for g_target, target in zip(g_targets, props['targets']):
                z_target = g_target.positions[:,2] - midz
                zetas[target] += list(z_target)
        elif ts.time > props['stop_ps']:
            break

    for target in props['targets']:
        zetas[target] = np.array(zetas[target])
        zetas[target] = zetas[target][np.logical_and(zetas[target] >= props['z_range'][0], zetas[target] <= props['z_range'][1])]
        counts, bins = np.histogram(zetas[target], bins=Z, density=True)
        distrs[target] = counts

    return ZC, distrs

def write_PCDistr(zetas, distr_dict, props):
    f = open(NAME + "_pcdistr.sfu", 'w')
    values = []
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    f.write("#space (nm) ")
    for key, val in distr_dict.items():
        values.append(val)
        f.write("{:<8} ".format(key))
    f.write("\n")
    values = np.array(values)*10 #A-1 to nm-1
    for i in range(len(zetas)):
        f.write("{:<8.3f} ".format(zetas[i]/10.)) #10 for A to nm
        for val in values:
            f.write("{:>8.3f} ".format(val[i]))
        f.write("\n")
    f.close()


zetas, distrs = PCDistribution(props_PCDistr)
write_PCDistr(zetas, distrs, props_PCDistr)
