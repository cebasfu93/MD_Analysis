"""
Calculates the average lipid order parameter from the Ci - Ci+1 vector (averaged for all bead-pairs and all lipids from a group in a frame).
"""
XTC = "NP61-POPC6-46_PRO1_FIX.xtc"
TPR = "NP61-POPC6-46_PRO1.tpr"
NAME = "NP61-POPC6-46_PRO1"
N_proc = 12
UP = False
DOWN = True
pairs = [#['GL2', 'GL1'], ['GL1', 'PO4'], ['PO4', 'NC3'],
          ['C1A', 'GL1'], ['D2A', 'C1A'], ['C3A', 'D2A'], ['C4A', 'C3A'],
          ['C1B', 'GL2'], ['C2B', 'C1B'], ['C3B', 'C2B'], ['C4B', 'C3B']]

from MDAnalysis import *
import numpy as np
import multiprocessing

U = Universe(TPR, XTC)
sel = {
'NP610' : U.select_atoms("name Pt"),
'MEM' : U.select_atoms("resname POPC"),
"LocalPO4" : U.select_atoms("(same residue as (cyzone 35 60.0 -60.0 (name Pt BB SC1 SC2 SC3 SC4 SCP SCN))) and name PO4", updating = True)
}

def leafletize(selection, up = UP, down = DOWN):
    Z_mean = np.mean(selection.positions[:,2])
    upper = selection.select_atoms("same residue as (name PO4 and prop z >= {})".format(Z_mean), updating = True)
    lower = selection.select_atoms("same residue as (name PO4 and prop z < {})".format(Z_mean), updating = True)
    if UP:
        leaflet = upper
    elif DOWN:
        leaflet = lower
    return leaflet

def order_one_frame(selection):
    leaflet = leafletize(selection)
    if UP:
        ax = [0, 0, 1]
    elif DOWN:
        ax = [0, 0, -1]

    ords = []
    for i in range(len(leaflet.residues)):
        for pair in pairs:
            a1 = leaflet.residues[i].atoms.select_atoms('name {}'.format(pair[0])).positions
            a2 = leaflet.residues[i].atoms.select_atoms('name {}'.format(pair[1])).positions
            a = np.subtract(a2, a1)
            cose = np.dot(a, ax)/np.linalg.norm(a)
            ords.append(1.5*cose**2-0.5)
    return np.mean(ords), np.std(ords)

def order_one_frame_parallel(coordinates):
    ords = []
    if UP:
        ax = [0, 0, 1]
    elif DOWN:
        ax = [0, 0, -1]

    for i in range(len(coordinates)):
        a1 = coordinates[i,0]
        a2 = coordinates[i,1]
        a = np.subtract(a2, a1)
        cose = np.dot(a, ax)/np.linalg.norm(a)
        ords.append(1.5*cose**2-0.5)
    return np.mean(ords), np.std(ords)

def order_over_time(uni, select, key, start_ns, stop_ns):
    start_ps = start_ns * 1000
    stop_ps = stop_ns * 1000
    ord_ave = []
    ord_std = []
    times = []
    all_coords = []

    for ts in uni.trajectory:
        t = ts.time
        if t >= start_ps and t <= stop_ps:
            times.append(t)
            leaflet = leafletize(select[key])
            n_r = len(leaflet.residues)
            n_p = len(pairs)
            coords_pairs = np.zeros((n_r*n_p, 2, 3))
            for i in range(n_r):
                for j, pair in enumerate(pairs):
                    a1 = leaflet.residues[i].atoms.select_atoms("name {}".format(pair[0])).positions
                    a2 = leaflet.residues[i].atoms.select_atoms("name {}".format(pair[1])).positions
                    coords_pairs[i*n_p+j, 0, :] = a1
                    coords_pairs[i*n_p+j, 1, :] = a2
            all_coords.append(coords_pairs)

    pool = multiprocessing.Pool(processes = N_proc)
    lop = pool.map(order_one_frame_parallel, all_coords)
    for i in range(len(lop)):
        ord_ave.append(lop[i][0])
        ord_std.append(lop[i][1])
    times = np.array(times)/1000

    f = open(NAME + "_lop_" + key + ".sfu", "w")
    f.write("#Lipid order parameter (lop)\n")
    f.write("#lop calculated for: {}\n#Start (ns): {}\n#Stop (ns): {}\n#Leaflet (up): {}\n#Leaflet (down): {}\n".format(key, start_ns, stop_ns, UP, DOWN))
    f.write("#lop (mean of means +/- deviation of means):\n#{:.5f} +/- {:.5f}\n".format(np.mean(ord_ave), np.std(ord_ave)))
    f.write("#Time (ns)\tMean lop (nm2)\tDeviation lop (nm2)\n")
    for i in range(len(ord_ave)):
        f.write("{:.3f}\t{:.5f}\t{:.5f}\n".format(times[i], ord_ave[i], ord_std[i]))
    f.close()

order_over_time(U, sel, 'LocalPO4', 900, 1000)
