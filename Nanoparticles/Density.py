#CALCULATES THE RADIAL MASS DENSITY OF GIVEN GROUPS
XTC = "NP18-53_PRO1_FIX.xtc"
TPR = "NP18-53_PRO1.tpr"
NAME = XTC[:-8]

import math
import numpy as np
import matplotlib.pyplot as plt
from Extras import *
from MDAnalysis import *
plt.rcParams["font.family"] = "Times New Roman"

U = Universe(TPR, XTC)
DT = U.trajectory[0].dt
sel = {
"all_gold" : U.select_atoms("name AU AUS AUL"),
"NP18"     : U.select_atoms("name AU AUS AUL or resname L18"),
"ST"     : U.select_atoms("name ST"),
"L18"     : U.select_atoms("resname L18"),
"C9"       : U.select_atoms("resname L18 and name C9"),
"NA"       : U.select_atoms("resname NA"),
"CL"       : U.select_atoms("resname CL"),
"SOL"      : U.select_atoms("resname SOL"),
"not-SOL"  : U.select_atoms("not resname SOL")
}

props_dens = {
'ref'       : "all_gold",
'targets'   : ["ST", "C9", "L18", "NA", "CL", "SOL"],
'start_ps'  : 25000,
'stop_ps'   : 100000,
'r_range'   : (0, 30),
'nbins'     : 150,
'dt'        : 10
}

def radial_density(props):
    densities = {}
    R = np.linspace(props['r_range'][0], props['r_range'][1], props['nbins'])
    n_frames = int((props['stop_ps'] - props['start_ps'])/props['dt'] + 1)
    V = 4/3 * math.pi * np.subtract(np.power(R[1:],3), np.power(R[:-1], 3))
    RC = center_bins(R)
    g_ref = sel[props['ref']]
    print("Reference COM: {}".format(props['ref']))
    for target in props['targets']:
        print("Current target: {}".format(target))
        M = []
        g_target = sel[target]
        for ts in U.trajectory:
            if ts.time >= props['start_ps'] and ts.time <= props['stop_ps'] and ts.time%props['dt'] == 0:
                dist = np.linalg.norm(np.subtract(g_target.positions, g_ref.center_of_mass()), axis = 1)
                for r1, r2 in zip(R[:-1], R[1:]):
                    m = np.sum(g_target.masses[np.logical_and(dist >= r1, dist  < r2)])
                    M.append(m)
        M = np.mean(np.array(M).reshape((n_frames, len(RC))), axis = 0)
        densities[target] = np.divide(M, V)
    return RC, densities

def write_density(space, dens_dict, properties):
    f = open(NAME + "_dens.sfu", 'w')
    values = []
    f.write("#Densities in kg/m3\n")
    for key, val in properties.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    f.write("#space (nm) ")
    for key, val in dens_dict.items():
        values.append(val)
        f.write("{:<8} ".format(key))
    f.write("\n")
    values = np.array(values)
    for i in range(len(space)):
        f.write("{:<8.3f} ".format(space[i]/10.)) #10 for A to nm
        for val in values:
            f.write("{:>8.3f} ".format(val[i] * 1660.539)) #1660.539 for uma/A3 to kg/m3
        f.write("\n")
    f.close()

r, densities = radial_density(props_dens)
write_density(r, densities, props_dens)

fig = plt.figure()
ax = plt.axes()
for target, density in densities.items():
    ax.plot(r/10, density*1660.539, label = target)
plt.legend()
plt.show()
