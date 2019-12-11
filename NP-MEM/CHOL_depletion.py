#PLOTS THE NUMBER OF CHOLESTEROL RESIDUES AT DIFFERENT XY LOCATIONS IN THE MEMBRANE
XTC = "NP610-POPC6-46_PRO1_FIX.xtc"
TPR = "NP610-POPC6-46_PRO1.tpr"
NAME = "NP610-POPC6-46_PRO1"
Z = 26
N_res = 30

import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams
from matplotlib import rc
#rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
rc('text', usetex=True)

U = mda.Universe(TPR, XTC)
deltat = U.trajectory.dt
sel = {
'NP610': U.select_atoms("name Pt", updating = True),
'CHOL': U.select_atoms("resname CHOL", updating = True),
'POPC': U.select_atoms("resname POPC", updating = True)
}

def t2f(time):
    return int(round(time*1000/deltat))

def plot_chol_dep(uni, selelect, start_ns, stop_ns):
    start_ps = start_ns * 1000
    stop_ps = stop_ns * 1000
    space = np.zeros((N_res, N_res))
    for ts in uni.trajectory[t2f(start_ns):t2f(stop_ns)]:
        PO4 = uni.select_atoms("name PO4")
        PO4_xy = PO4.positions[:,:2]
        dx = (np.max(PO4_xy[:,0]) - np.min(PO4_xy[:,0]))/N_res
        dy = (np.max(PO4_xy[:,1]) - np.min(PO4_xy[:,1]))/N_res
        chol = uni.select_atoms("resname CHOL")
        for c in chol.residues:
            cxy = c.atoms.centroid()[:2]
            x_ndx = int(round(cxy[0]/dx)%N_res)
            y_ndx = int(round(cxy[1]/dy)%N_res)
            space[y_ndx, x_ndx] +=1

    DX = dx * N_res
    DY = dy * N_res
    tx = np.linspace(0, N_res-1, 5)
    txl = [-6.5, -3.2, 0, 3.2, 6.5] #np.linspace(-DX/2, DX/2, 5).astype('int')
    ty = np.linspace(0, N_res-1, 5)
    tyl = [-6.5, -3.2, 0, 3.2, 6.5] #np.linspace(-DY/2, DY/2, 5).astype('int')

    fig = plt.figure()
    ax = plt.axes()
    ax.set_xticks(tx)
    ax.set_xticklabels(txl, fontsize = Z, fontname = 'Arial')
    ax.set_yticks(ty)
    ax.set_yticklabels(tyl, fontsize = Z, fontname = 'Arial')
    ax.set_xlabel('X (nm)', fontsize = Z)
    ax.set_ylabel('Y (nm)', fontsize = Z)
    cax = ax.imshow(space, interpolation = 'bilinear', cmap = cm.RdBu_r, vmin = 100, vmax = 1100)
    cbar = fig.colorbar(cax, ticks = [200, 400, 600, 800, 1000], ax = ax)
    axc = cbar.ax
    axc.set_yticklabels([200, 400, 600, 800, 1000], fontsize = Z, fontname = 'Arial')
    cbar.set_label("Cholesterol Count", fontsize = Z, rotation = 270, labelpad = 27)
    plt.tight_layout()
    plt.savefig(NAME + "_choldep.png", dpi = 300)
    plt.show()
    plt.close()

plot_chol_dep(U, sel, 250, 350)
