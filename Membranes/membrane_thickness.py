#CALCULATES THE MEMBRANE THICKNESS

XTC = "POPC2-24_PRO1-2_FIX.xtc"
TPR = "POPC2-24_PRO1.tpr"
NAME = XTC[:-8]

from MDAnalysis import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi, voronoi_plot_2d
import multiprocessing

U = Universe(TPR, XTC)
sel = {
"P31" : U.select_atoms("name P31")
#"LocalPO4" : U.select_atoms("(same residue as (cyzone 35 60.0 -60.0 (name Pt BB SC1 SC2 SC3 SC4 SCP SCN))) and name PO4", updating = True)
}

props_thick={
"ref"       : 'P31',
"start_ps"  : 25000,
"stop_ps"   : 100000,
"dt"        : 80,
"n_proc"    : 12,
"resolution": 40,
"n_neigh"   : 3
}

def get_thickness_one_frame(coords_mem):
    Z_mean = np.mean(coords_mem[:,2])
    upper = coords_mem[coords_mem[:,2] > Z_mean,:]
    lower = coords_mem[coords_mem[:,2] < Z_mean,:]

    x = np.linspace(np.min(coords_mem[:,0]), np.max(coords_mem[:,0]), props_thick["resolution"] + 1)
    dx = x[1] - x[0]
    y = np.linspace(np.min(coords_mem[:,1]), np.max(coords_mem[:,1]), props_thick["resolution"] + 1)
    dy = y[1] - y[0]
    thickness = np.zeros((props_thick["resolution"], props_thick["resolution"]))
    for i in range(props_thick["resolution"]):
        for j in range(props_thick["resolution"]):
            dist_upper = cdist([[x[i]+0.5*dx, y[j]+0.5*dy]], upper[:,:2])
            dist_lower = cdist([[x[i]+0.5*dx, y[j]+0.5*dy]], lower[:,:2])
            close_upper = np.argsort(dist_upper[0])[:props_thick["n_neigh"]]
            close_lower = np.argsort(dist_lower[0])[:props_thick["n_neigh"]]
            av_upper = np.mean(upper[close_upper][:,2])
            av_lower = np.mean(lower[close_lower][:,2])
            thickness[j, i] = av_upper - av_lower
    return thickness

def get_thickness_over_time(props):
    times = []
    all_coords = []
    tk_ave = []
    tk_std = []
    g_ref = sel[props['ref']]

    for ts in U.trajectory:
        if ts.time >= props["start_ps"] and ts.time%props['dt']==0:
            times.append(ts.time)
            all_coords.append(g_ref.positions)
        if ts.time >= props["stop_ps"]:
            break

    pool = multiprocessing.Pool(processes = props["n_proc"])
    tk = pool.map(get_thickness_one_frame, all_coords)
    tk = np.array(tk)/10 #10 A to nm
    times = np.array(times)
    return times, tk

def write_thickness(times, thick, props):
    f = open(NAME+"_thick.sfu", "w")
    f.write("#Membrane thickness\n")
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    f.write("#Membrane thickness (mean of all values +/- deviation of all values): {:.3f} +/- {:.3f} nm\n".format(np.mean(thick), np.std(thick)))
    f.write("#Thickness (nm)\n")
    for t, tks in enumerate(thick):
        f.write("#T - > {:<10.3f} ps\n".format(times[t]))
        for i in range(props["resolution"]):
            for j in range(props["resolution"]):
                f.write("{:<8.3f} ".format(tks[i,j]))
            f.write("\n")
    f.close()

time, thickness = get_thickness_over_time(props_thick)
write_thickness(time, thickness, props_thick)
