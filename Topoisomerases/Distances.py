"""
Calculates distance between COM of specificied groups. 
Not generalized yet. 
Specific for Topoisomerase 2alpha complexed with ARN compound.
"""
XTC = "SYS-1_PRO1-5_VIS.xtc" #D has the NP centered
TPR = "SYS-1_PRO1.tpr"
NAME = XTC[:-8]

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from MDAnalysis import *
plt.rcParams["font.family"] = "Arial"

U = Universe(TPR, XTC)
DT = U.trajectory[0].dt

sel = {
"Carg"         : U.select_atoms("name CZ and resid 790"),
"Ering"        : U.select_atoms("resname A22 and name C1 C2 C3 C4 C5 C6"),
"G5base"       : U.select_atoms("resid 1470 and name N1 C2 N3 C4 C5 C6 N7 C8 N9"),
"Barbring"     : U.select_atoms("resname A22 and name C8 C9 C10 C11 N2 N3"),
"C1base"       : U.select_atoms("resid 1499 and name N1 C2 N3 C4 C5 C6"),
}

times = []
d_Carg_Ering = []
d_G5base_barbring = []
d_Carg_Ering_min = []
d_G5base_barbring_min = []
d_C1base_barbring = []
d_C1base_barbring_min = []

def calc_dist(key1, key2):
    g1 = sel[key1]
    g2 = sel[key2]
    d = np.linalg.norm(g1.center_of_mass() - g2.center_of_mass())
    return d

def calc_min_dist(key1, key2):
    g1 = sel[key1]
    g2 = sel[key2]
    dists = cdist(g1.positions, g2.positions)
    d = np.min(dists)
    return d

for ts in U.trajectory:
    if ts.time%10000 == 0:
        print(ts.time)
    times.append(ts.time)
    d1 = calc_dist('Ering', 'Carg')
    d_Carg_Ering.append(d1)
    d2 = calc_dist('Barbring', 'G5base')
    d_G5base_barbring.append(d2)
    d3 = calc_dist('Barbring', 'C1base')
    d_C1base_barbring.append(d3)
    d4 = calc_min_dist('Ering', 'Carg')
    d_Carg_Ering_min.append(d4)
    d5 = calc_min_dist('Barbring', 'G5base')
    d_G5base_barbring_min.append(d5)
    d6 = calc_min_dist('Barbring', 'C1base')
    d_C1base_barbring_min.append(d6)

times = np.array(times)/1000
d_Carg_Ering = np.array(d_Carg_Ering)/10
d_G5base_barbring = np.array(d_G5base_barbring)/10
d_C1base_barbring = np.array(d_C1base_barbring)/10
d_Carg_Ering_min = np.array(d_Carg_Ering_min)/10
d_G5base_barbring_min = np.array(d_G5base_barbring_min)/10
d_C1base_barbring_min = np.array(d_C1base_barbring_min)/10

f = open(NAME+"_distances.sfu", "w")
f.write("#d1: Carg - E_ring\n")
f.write("#d2: G5base - Barbring\n")
f.write("#d3: C1base - Barbring\n")
f.write("#d4: Min(Carg - E_ring)\n")
f.write("#d5: Min(G5base - Barbring)\n")
f.write("#d6: Min(C1base - Barbring)\n")
f.write("#{:<10} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}\n".format("Time (ns)", "d1 (nm)", "d2 (nm)", "d3 (nm)", "d4 (nm)", "d5 (nm)", "d6 (nm)"))
for t, d1, d2, d3, d4, d5, d6 in zip(times, d_Carg_Ering, d_G5base_barbring, d_C1base_barbring, d_Carg_Ering_min, d_G5base_barbring_min, d_C1base_barbring_min):
    f.write("{:<10.3f} {:<8.3f} {:<8.3f} {:<8.3f} {:<8.3f} {:<8.3f} {:<8.3f}\n".format(t, d1, d2, d3, d4, d5, d6))
f.close()
