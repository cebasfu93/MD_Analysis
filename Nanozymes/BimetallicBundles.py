XTC = "3_production_LIG2_SUB_small.xtc"
TPR = "LIG2_PROD_fake.tpr"
NAME = XTC[:-4]

import numpy as np
import matplotlib.pyplot as plt
from MDAnalysis import *
from scipy.spatial.distance import cdist
from itertools import zip_longest

U = Universe(TPR, XTC)
sel = {
"all_gold"    : U.select_atoms("name AU AUS AUL"),
"Zn"          : U.select_atoms("name Zn")
}

DT = U.trajectory[0].dt
print(DT)

props = {
'd_thresh'  : 5,
'start_ps'  : 0, #123
'stop_ps'   : 1000000,# 567
}

g_au  = sel['all_gold']
g_zn  = sel['Zn']

flatten_list = lambda l: [item for sublist in l for item in sublist]

def zn_matrix():
    matrix = []
    dists = []
    ndx_ini = int(np.round(max((props['start_ps'] - U.trajectory[0].time)/DT, 0), 0))
    ndx_fin = int(np.round((props['stop_ps'] - U.trajectory[0].time)/DT,0))
    for ts in U.trajectory[ndx_ini:ndx_fin]:
        if ts.time >= props['start_ps'] and ts.time <= props['stop_ps']:
            print(ts.time, end='\r')
            M = cdist(g_zn.positions, g_zn.positions)
            XX1, XX2 = np.meshgrid(g_zn.positions[:,0], g_zn.positions[:,0])
            YY1, YY2 = np.meshgrid(g_zn.positions[:,1], g_zn.positions[:,1])
            ZZ1, ZZ2 = np.meshgrid(g_zn.positions[:,2], g_zn.positions[:,2])
            D = np.linalg.norm(np.swapaxes(np.array([XX1+XX2, YY1+YY2, ZZ1+ZZ2]), 0, 2) - g_au.center_of_mass(), axis=2)
            matrix.append(M)
            dists.append(D)
        elif ts.time >= props['stop_ps']:
            break

    matrix = np.array(matrix)
    dists = np.array(dists)

    return matrix, dists

def process():
    zn_dists, com_dists = zn_matrix()
    triu_mask = np.triu_indices(g_zn.n_atoms)
    zn_dists[:,triu_mask[0], triu_mask[1]] = 1000
    zn_cond = zn_dists <= props['d_thresh']
    zn_cond = zn_cond.reshape((len(zn_dists), g_zn.n_atoms**2))
    com_dists = com_dists.reshape((len(zn_dists), g_zn.n_atoms**2))

    fracs, restimes, moments = {}, {}, {}
    fracs['ZnZn'] = calculate_occupancy(zn_cond)
    restimes['ZnZn'] = calculate_residence_time(zn_cond, com_dists)
    moments['ZnZn'] = statistical_moments(restimes['ZnZn'][0])
    return fracs, restimes, moments

def fix_switch_list(flat_switch):
    ndx_ini = int(np.round(max((props['start_ps'] - U.trajectory[0].time)/DT, 0), 0))

    fixed = [i + ndx_ini for i in flat_switch]
    return fixed

def calculate_residence_time(onoff_raw, dists_p):
    onoff = np.zeros((onoff_raw.shape[0]+2, onoff_raw.shape[1]))
    onoff[1:-1] = onoff_raw
    gates = onoff[1:,:] - onoff[:-1,:]
    on_switch, off_switch = [list(np.where(gate==1)[0]+1) for gate in gates.T], [list(np.where(gate==-1)[0]) for gate in gates.T]

    b_res1 = [[g_zn.residues.ix[i%g_zn.n_atoms]+1]*len(on) for i, on in enumerate(on_switch)]
    b_res1 = flatten_list(b_res1)

    b_res2 = [[g_zn.residues.ix[i//g_zn.n_atoms]+1]*len(on) for i, on in enumerate(on_switch)]
    b_res2 = flatten_list(b_res2)

    b_times = [[(b-a+1)*DT for a,b in zip(on, off)] for on, off in zip(on_switch, off_switch)]
    b_times = flatten_list(b_times)

    median_dists = [[np.round(np.median(dists_p[a-1:b,i])/10, 2) for a,b in zip(on,off)] for i,(on,off) in enumerate(zip(on_switch, off_switch))]
    median_dists = flatten_list(median_dists)

    on_switch, off_switch = flatten_list(on_switch), flatten_list(off_switch)
    on_switch, off_switch = fix_switch_list(on_switch), fix_switch_list(off_switch)

    return b_times, median_dists, on_switch, off_switch, b_res1, b_res2

def bootstrap(onoff, boot_iter=10000, boot_size=0.01):
    np.random.seed(666)
    pts_ndxs = np.linspace(0, len(onoff)-1, len(onoff), dtype='int')
    boots = []
    for i in range(boot_iter):
        boot_ndxs = np.random.choice(pts_ndxs, size=int(boot_size*len(onoff)), replace=True)
        boot_pts = onoff[boot_ndxs]
        boots.append(np.mean(boot_pts))

    return np.mean(boots), np.std(boots)

def calculate_occupancy(onoff_raw):
    onoff = onoff_raw.flatten()
    frac_ave, frac_std = bootstrap(onoff)
    return frac_ave*100, frac_std*100

def statistical_moments(btimes):
    if len(btimes) != 0:
        hist, bins = np.histogram(btimes, density=True, range=(0,5000), bins=100)
        bins = (bins[1:] + bins[:-1])*0.5
        dx = bins[1] - bins[0]
        first = np.sum(np.multiply(hist*dx, bins))
        second = np.sum(np.multiply(hist*dx, bins**2)) - first**2
        second = np.sqrt(second)
    else:
        first, second = -1, -1
    return first, second

def write_output(fractions, residence_times, moments):
    f = open(NAME+"_restimes_mono.sfu", "w")
    #Writes input parameters
    f.write("#Residence time (ps) for bound, precatalytic, precatalytic_solvated, and active complexes for mechanism 1,2, and 3, and the minimum distance between phosphorus and Au C.O.M.\n")
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    #Writes percental occupations
    f.write("\n#Mean and standard deviation of percentual occupations\n")
    for key, val in fractions.items():
        f.write("#{:<8.3f} +/- {:>8.3f} ({})\n".format(*val, key))
    #Writes statistical moments
    f.write("\n#First and second statistical moments of residence times distributions\n")
    for key, val in moments.items():
        f.write("#{:<8.3f} +/- {:>8.3f} ({})\n".format(*val, key))
    #Writes residence times
    for key, val in residence_times.items():
        f.write("\n#@@@{}\n".format(key))
        f.write("#{:<7}(ps) {:<8}(nm) {:<8}(fr) {:<8}(fr) {:<8}(id) {:<8}(id)\n".format("Restime", "DistP", "Start", "End", "Resid1", "Resid2"))
        for pt in zip(*val):
            f.write(("{:<12} "*len(pt)).format(*pt) + "\n")
    f.close()

fractions, residence_times, moments = process()
write_output(fractions, residence_times, moments)
