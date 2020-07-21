XTC = "NP22sp-53_PRO1-10_FIX.xtc"
TPR = "NP22sp-53_PRO1.tpr"
NAME = XTC[:-8]

import math
import numpy as np
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist
from Extras import *
from MDAnalysis import *

U = Universe(TPR, XTC)
DT = U.trajectory[0].dt
sel = {
"all_gold"           : U.select_atoms("name AU AUS AUL"),
"S_sulfonate"        : U.select_atoms("resname L22 and name S1"),
"SER"                : U.select_atoms("resname SER"),
"SER_Nterminal"      : U.select_atoms("resname SER and name N1"),
"SER_Ncycle"         : U.select_atoms("resname SER and name N2"),
"SER_O"              : U.select_atoms("resname SER and name O1"),
"PHE"                : U.select_atoms("resname PHE"),
"PHE_N"              : U.select_atoms("resname PHE and name N1"),
"PHE_Cacid"          : U.select_atoms("resname PHE and name C9"),
"PHE_Ccycle"         : U.select_atoms("resname PHE and name C5"),
"NA"                 : U.select_atoms("resname NA"),
"SOL"                : U.select_atoms("resname SOL"),
}

props_pair = {
'ref'       : "S_sulfonate",
'targets'   : ["SER_Nterminal", "SER_Ncycle", "SER_O", "PHE_N", "PHE_Cacid", "PHE_Ccycle", "NA", "SOL"],
'start_ps'  : 0,
'stop_ps'   : 1000000,
'r_range'   : (0, 20),
'nbins'     : 150,
'dt'        : 1
}

def pair_manual(props):
    pairs = {}
    n_frames = ps_frame(props['stop_ps'], DT) - ps_frame(props['start_ps'], DT) + 1
    R = np.linspace(props['r_range'][0], props['r_range'][1], props['nbins'])
    dr = R[1] - R[0]
    g_ref = sel[props['ref']]
    print("Reference group: {}".format(props['ref']))
    for target in props['targets']:
        print("Current target: {}".format(target))
        counts = np.zeros(len(R)-1)
        g_target = sel[target]
        for ts in U.trajectory:
            if ts.time >= props['start_ps'] and ts.time <= props['stop_ps'] and ts.time % props['dt'] == 0:
                x_ref = g_ref.positions
                x_target = g_target.positions
                dists = cdist(x_ref, x_target)
                dists = dists[dists<=props['r_range'][1]]
                counts += np.histogram(dists, bins=R)[0]
            elif ts.time > props['stop_ps']:
                break

        counts = counts/(n_frames*g_ref.n_atoms)
        n_target = np.sum(counts)
        norm = props['r_range'][1]**3 / (3*n_target * np.power(R[1:], 2) * dr)
        counts = np.multiply(counts, norm)

        pairs[target] = counts
    return R[1:], pairs

def write_pair(space, pair_dict, props):
    f = open(NAME + "_pair.sfu", 'w')
    values = []
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    f.write("#space (nm) ")
    for key, val in pair_dict.items():
        values.append(val)
        f.write("{:<8} ".format(key))
    f.write("\n")
    values = np.array(values)
    for i in range(len(space)):
        f.write("{:<8.3f} ".format(space[i]/10.)) #10 for A to nm
        for val in values:
            f.write("{:>8.3f} ".format(val[i]))
        f.write("\n")
    f.close()

r, pairs = pair_manual(props_pair)
write_pair(r, pairs, props_pair)

fig = plt.figure()
ax = plt.axes()
for target, pair in pairs.items():
    ax.plot(r, pair, label = target)
plt.legend()
plt.show()
