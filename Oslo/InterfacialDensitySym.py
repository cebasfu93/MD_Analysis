XTC = "T1-N40-I30_AA_PRO1-6_FIX.xtc"
TPR = "T1-N40-I30_AA_PRO1.tpr"
NAME = XTC[:-8]

import numpy as np
from scipy.spatial.distance import cdist
from MDAnalysis import *

U = Universe(TPR, XTC)
DT = U.trajectory[0].dt

sel = {
"NP"       : U.select_atoms("bynum 1:459"),
"cit_solv" : U.select_atoms("resname CIT NA CL SOL")
}

props_interfacial_density = {
'ref'       : "NP",
'target'    : "cit_solv",
'dx'        : 1, #A, grid spacing
'edgesize'  : 30, #A, size of the square grid
'start_ps'  : 0,
'stop_ps'   : 250000,
'dt'        : 20
}


def histogramize(props, data):
    n_b = int(props['edgesize']//props['dx'] + 1)
    b_bins = np.linspace(-props['edgesize']/2, props['edgesize']/2, n_b)

    digx = np.digitize(data[:,0], bins=b_bins)-1
    digy = np.digitize(data[:,1], bins=b_bins)-1
    digz = np.digitize(data[:,2], bins=b_bins)-1
    populations = np.zeros((n_b-1, n_b-1, n_b-1))
    counts = np.zeros_like(populations)
    for i, (nx, ny, nz) in enumerate(zip(digx, digy, digz)):
        populations[nx, ny, nz] += data[i, 3]
        counts[nx, ny, nz] += 1
    populations = np.divide(populations, counts, out=np.zeros_like(populations), where=counts!=0)

    return populations

def interfacial_density(props):
    n_frames = len(U.trajectory)
    g_ref = sel[props['ref']]
    g_target = sel[props['target']]

    populations = []
    for ts in U.trajectory:
        if ts.time >= props['start_ps'] and ts.time <= props['stop_ps'] and ts.time%props['dt']==0:
            if ts.time%50000 == 0:
                print(ts.time)
            data = np.array([[0,0,0,0]])
            b1, b2, b3 = np.identity(3)
            pts = g_target.positions - g_ref.center_of_mass()
            condition_b1 = np.logical_and(pts[:,0] >= -props['edgesize']/2, pts[:,0] <= props['edgesize']/2)
            condition_b2 = np.logical_and(pts[:,1] >= -props['edgesize']/2, pts[:,1] <= props['edgesize']/2)
            condition_b3 = np.logical_and(pts[:,2] >= -props['edgesize']/2, pts[:,2] <= props['edgesize']/2)
            conditions = np.logical_and(condition_b1, np.logical_and(condition_b2, condition_b3))
            pts_relevant = pts[conditions,:]
            charges = g_target.charges[conditions]
            coords_charge = np.append(pts_relevant, np.array([charges]).T, axis=1)
            data = np.vstack((data, coords_charge))
            pop = histogramize(props, data)
            populations.append(pop)
        elif ts.time > props['stop_ps']:
            break

    populations = np.mean(populations, axis=0)
    print(populations.shape)
    return populations

def write_populations(props, data):
    n_b = int(props['edgesize']//props['dx'] + 1)
    b_bins = np.linspace(-props['edgesize']/2, props['edgesize']/2, n_b)
    b_bins = (b_bins[:-1] + b_bins[1:])/2
    f = open(NAME+"_interdensitysym.sfu", "w")
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    f.write("Each slice is divided by $(CVvalue)")
    f.write("#Interfacial charge distribution (e)\n")
    for i, cv1 in zip(range(len(data[:,0,0])), b_bins):
        f.write("$ CV = {:<8.3f} nm\n".format(cv1/10)) #A to nm
        for j in range(len(data[0,:,0])):
            for k in range(len(data[0,0,:])):
                f.write("{:<8.3f} ".format(data[i,j,k]))
            f.write("\n")
    f.close()

data = interfacial_density(props_interfacial_density)
write_populations(props_interfacial_density, data)
