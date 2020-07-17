"""
Calculates the COM of cholesterol and it returns all the XYZ coordinates over time.
Useful for building 2D maps of cholesterol density.
"""
XTC = "NP61-POPC6-46_PRO1_FIX.xtc"
TPR = "NP61-POPC6-46_PRO1.tpr"
NAME = XTC[:-8]
Z = 26
N_res = 50
N_proc = 12

import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams
from matplotlib import rc
import multiprocessing
#rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
rc('text', usetex=True)

U = mda.Universe(TPR, XTC)
deltat = U.trajectory.dt
sel = {
'gH': U.select_atoms("name BB SC1 SC2 SC2 SC4 SCP SCN"),
'CHOL': U.select_atoms("resname CHOL"),
'PO4': U.select_atoms("name PO4")
#'POPC': U.select_atoms("resname POPC", updating = True)
}

def t2f(time):
    return int(round(time*1000/deltat))

def chol_dep_one_frame(coms_lx_ly):
    space = np.zeros((N_res, N_res))
    cxy, lx, ly = coms_lx_ly[0], coms_lx_ly[1], coms_lx_ly[2]
    dx = lx/N_res
    dy = ly/N_res
    x_ndxs = ((cxy[:,0]/dx).round(0)%N_res).astype('int')
    y_ndxs = ((cxy[:,1]/dy).round(0)%N_res).astype('int')
    space[y_ndxs, x_ndxs] += 1
    return space


def chol_dep_xy(uni, start_ns, stop_ns):
    g_chol = sel['CHOL']
    g_mol = sel['gH']
    g_head = sel['PO4']
    start_ps = start_ns * 1000
    stop_ps = stop_ns * 1000
    times = []
    coms = []
    for ts in uni.trajectory:
        if ts.time >= start_ps and ts.time <= stop_ps:
            times.append(ts.time)
            if ts.time%100000 == 0:
                print(ts.time)

            cxyz =[c.atoms.center_of_mass() for c in g_chol.residues]
            coms.append(cxyz)
        elif ts.time > stop_ps:
            break
    coms = np.array(coms)/10 #10A to nm
    return times, coms

def write_chol_dep_xy(times, coms):
    f = open(NAME+"_chol_xy.sfu", "w")
    f.write("#Cholesterol molecules centers of mass\n")
    f.write("#X (nm) \t Y (nm) \t Z (nm)\n")
    for t, com in zip(times, coms):
        f.write("#T - > {:<10.3f} ps\n".format(t))
        for c in com:
            f.write("{:<8.3f} {:<8.3f} {:<8.3f}\n".format(*c))
    f.close()

times, coms = chol_dep_xy(U, 0, 1000)
write_chol_dep_xy(times, coms)
