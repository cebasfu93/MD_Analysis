#CALCULATES THE DISTANCE BETWEEN THE NP AND THE MEMBRANE IN THE Z DIRECTION
XTC = "NP18-POPC2-54_PRO1-37_FIX_C.xtc" #Trajectory with membrane centered
TPR = "NP18-POPC2-54_PRO1.tpr"
NAME = XTC[:-10]

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
"P31" : U.select_atoms("name P31"),
"center_gold" : U.select_atoms("bynum 10 or bynum 37")
}

props_dz = {
'ref1'      : "center_gold",
'ref2'      : "P31",
'start_ps'  : 0,
'stop_ps'   : 1000000,
'dt'        : 10
}


def distanceZ(props):
    g_ref1 = sel[props['ref1']]
    g_ref2 = sel[props['ref2']]
    dz = []
    times = []
    for ts in U.trajectory:
        if ts.time >= props['start_ps'] and ts.time%props['dt']==0:
            d_temp = cdist(g_ref1.positions, g_ref1.positions)
            if np.all(d_temp < 10):
                z_ref1 = g_ref1.center_of_mass()[2]
                z_ref2 = g_ref2.center_of_mass()[2]
                dz.append(abs(z_ref1 - z_ref2))
                times.append(ts.time)
        if ts.time> props['stop_ps']:
            break
    dz = np.array(dz)
    times = np.array(times)
    return times, dz

def write_dz(times, dist, props):
    f = open(NAME+"_dz.sfu", "w")
    f.write("#Distance along Z (nm)\n")
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    f.write("#time (ps)   Z-Distance (nm)\n")

    for i in range(len(times)):
        f.write("{:<9.2f} ".format(times[i]))
        f.write("{:>9.2f}\n".format(dist[i]))
    f.close()

times, z = distanceZ(props_dz)
write_dz(times, z, props_dz)

fig = plt.figure()
ax = plt.axes()
ax.plot(times, z)
plt.show()
