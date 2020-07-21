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
from MDAnalysis.analysis.rdf import InterRDF
plt.rcParams["font.family"] = "Times New Roman"

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

props_rdf = {
'ref'       : "all_gold",
'targets'    : ["S_sulfonate", "SER", "SER_Nterminal", "SER_Ncycle", "SER_O", "PHE", "PHE_N", "PHE_Cacid", "PHE_Ccycle", "NA", "SOL"],
'start_ps'  : 0,
'stop_ps'   : 1000,
'r_range'   : (0, 35),
'nbins'     : 150,
'dt'        : 1
}

def rdf_manual(props):
    rdfs = {}
    rdfs_cum = {}
    n_frames = ps_frame(props['stop_ps'], DT) - ps_frame(props['start_ps'], DT) + 1
    R = np.linspace(props['r_range'][0], props['r_range'][1], props['nbins'])
    dr = R[1] - R[0]
    g_ref = sel[props['ref']]
    print("Reference COM: {}".format(props['ref']))
    for target in props['targets']:
        print("Current target: {}".format(target))
        counts = np.zeros(len(R)-1)
        g_target = sel[target]
        for ts in U.trajectory:
            if ts.time >= props['start_ps'] and ts.time <= props['stop_ps'] and ts.time % props['dt'] == 0:
                x_ref = g_ref.center_of_mass()
                x_target = g_target.positions
                dx = np.subtract(x_target, x_ref)
                dists = np.linalg.norm(dx, axis = 1)
                counts += np.histogram(dists, bins = R)[0]
            elif ts.time > props['stop_ps']:
                break

        counts = counts/n_frames
        n_target = np.sum(counts)
        norm = props['r_range'][1]**3 / (3*n_target * np.power(R[1:], 2) * dr)
        counts = np.multiply(counts, norm)

        rdfs[target] = counts
        homo_dens = 3*g_target.n_atoms/(4*np.pi*props['r_range'][1]**3)
        integrand = 4*np.pi*np.multiply(np.power(R[1:], 2), counts)
        cumulative = homo_dens*cumtrapz(y=integrand, x=R[1:], initial=0.0) #returns a shape with one element less
        rdfs_cum[target] = cumulative

    return R[1:], rdfs, rdfs_cum

def write_rdf(space, rdf_dict, properties):
    f = open(NAME + "_rdf.sfu", 'w')
    values = []
    for key, val in properties.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    f.write("#space(nm) ")
    for key, val in rdf_dict.items():
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

def write_rdf_cum(space, rdf_dict, properties):
    f = open(NAME + "_rdf_cum.sfu", 'w')
    values = []
    for key, val in properties.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    f.write("#space (nm) ")
    for key, val in rdf_dict.items():
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

r, rdfs, rdfs_cum = rdf_manual(props_rdf)
write_rdf(r, rdfs, props_rdf)
write_rdf_cum(r, rdfs_cum, props_rdf)

fig = plt.figure()
ax = plt.axes()
for target, rdf in rdfs.items():
    ax.plot(r, rdf, label = target)
plt.legend()
plt.show()

fig = plt.figure()
ax = plt.axes()
for target, rdf in rdfs_cum.items():
    ax.plot(r, rdf, label = target)
plt.legend()
plt.show()
