#RADIAL ENCLOSED CHARGE CONTRIBUTED BY THE VARIOUS COMPONENTS OF THE SYSTEM
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

props_charge = {
'ref'       : "all_gold",
'targets'   : ["all_gold", "L18", "NA", "CL", "SOL"], #Should add to all the system
'start_ps'  : 25000,
'stop_ps'   : 100000,
'r_range'   : (0, 30),
'nbins'     : 150,
'dt'        : 10
}

def charge(props):
    charges = {}
    R = np.linspace(props['r_range'][0], props['r_range'][1], props['nbins'])
    RC = center_bins(R)
    n_frames = int((props['stop_ps'] - props['start_ps'])/props['dt'] + 1)

    g_ref = sel[props['ref']]
    print("Reference COM: {}".format(props['ref']))

    for target in props['targets']:
        print("Current target: {}".format(target))
        Q = []
        g_target = sel[target]
        for ts in U.trajectory:
            if ts.time >= props['start_ps'] and ts.time <= props['stop_ps'] and ts.time%props['dt'] == 0:
                dist = np.linalg.norm(np.subtract(g_target.positions, g_ref.center_of_mass()), axis = 1)
                for r1, r2 in zip(R[:-1], R[1:]):
                    q = np.sum(g_target.charges[np.logical_and(dist >= r1, dist  < r2)])
                    Q.append(q)
        Q = np.mean(np.array(Q).reshape((n_frames, len(RC))), axis = 0)
        Q = np.cumsum(Q)
        charges[target] = Q
    Q_tot = []
    for key, val in charges.items():
        Q_tot.append(val)
    Q_tot = np.array(Q_tot)
    Q_tot = np.sum(Q_tot, axis = 0)
    charges["TOTAL"] = Q_tot
    return RC, charges

def write_charge(space, charge_dict, properties):
    f = open(NAME + "_charge.sfu", 'w')
    values = []
    f.write("#Charges in (e)\n")
    for key, val in properties.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    f.write("#space (nm) ")
    for key, val in charge_dict.items():
        values.append(val)
        f.write("{:<8} ".format(key))
    f.write("\n")
    values = np.array(values)
    for i in range(len(space)):
        f.write("{:<8.3f} ".format(space[i]/10.)) #10 for A to nm
        for val in values:
            f.write("{:>8.3f} ".format(val[i])) #1660.539 for uma/A3 to kg/m3
        f.write("\n")
    f.close()


r, charges = charge(props_charge)
write_charge(r, charges, props_charge)

fig = plt.figure()
ax = plt.axes()
for target, charge in charges.items():
    ax.plot(r/10, charge, label = target)
plt.legend()
plt.show()
