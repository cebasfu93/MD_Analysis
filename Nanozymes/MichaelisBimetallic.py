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
"SUB"         : U.select_atoms("resname SUB"),
"all_gold"    : U.select_atoms("name AU AUS AUL"),
"WAT_OW"      : U.select_atoms("resname SOL and name OW"),
"Zn"          : U.select_atoms("name Zn")
}

DT = U.trajectory[0].dt
print(DT)

props = {
'd1_thresh' : 2.5,#2.5,
'd3_thresh' : 2.5,
'd4_thresh' : 2.3,
'd5_thresh' : 2.0,
'start_ps'  : 0, #123
'stop_ps'   : 1000000,# 567
}

g_ow  = sel['WAT_OW']
g_au  = sel['all_gold']
g_zn  = sel['Zn']
g_sub = sel['SUB']

flatten_list = lambda l: [item for sublist in l for item in sublist]

class Substrate:
    def __init__(self, residue):
        ag = residue.atoms
        o3 = ag[ag.names=="O3"]
        o4 = ag[ag.names=="O4"]
        o5 = ag[ag.names=="O5"]
        ph = ag[ag.names=='P1']

        self.dzn = -100
        C1, g_zn_complex = self.evaluate_condition1(o4, o5)
        if C1:
            C2 = self.evaluate_condition2(o3, g_zn_complex)
            if C2:
                C3, g_o_complex = self.evaluate_condition3(g_zn_complex)
                C4 = self.evaluate_condition4(g_o_complex, o3)
            else:
                C3, C4 = False, False
        else:
            C2, C3, C4 = False, False, False

        self.bound = C1
        self.precat = C1 and C2
        self.precat_solv = C1 and C2 and C3
        self.active = C1 and C2 and C3 and C4

        self.d6 = np.linalg.norm(ph.positions[0] - g_au.center_of_mass())

    def evaluate_condition1(self, o4, o5):
        value = False
        all_d1 = np.linalg.norm(g_zn.positions - o5.positions[0], axis=1)
        cond_d1 = all_d1 <= props['d1_thresh']
        g_zn_d1 = U.atoms[g_zn[np.where(cond_d1)].ix]

        all_d2 = np.linalg.norm(g_zn.positions - o4.positions[0], axis=1)
        cond_d2 = all_d2 <= props['d1_thresh']
        g_zn_d2 = U.atoms[g_zn[np.where(cond_d2)].ix]

        if len(g_zn_d1) > 0 and len(g_zn_d2) > 0 and len(g_zn_d1.union(g_zn_d2)) > 1:
            all_d_zn = cdist(g_zn_d1.positions, g_zn_d2.positions)
            mask = all_d_zn > 0.0001
            self.dzn = np.min(all_d_zn[mask])

            value = True

        return value, g_zn_d1.union(g_zn_d2)

    def evaluate_condition2(self, o3, zn_complex):
        value = False
        all_d3 = np.linalg.norm(zn_complex.positions - o3.positions[0], axis=1)
        if np.any(all_d3 <= props['d3_thresh']):
            value = True
        return value

    def evaluate_condition3(self, zn_complex):
        value = False
        all_d4 = cdist(zn_complex.positions, g_ow.positions)
        cond4 = all_d4 <= props['d4_thresh']
        o_complex = g_ow[np.where(cond4)[1]]
        if np.any(cond4):
            value = True
        return value, o_complex

    def evaluate_condition4(self, o_complex, o3):
        value = False
        all_d5 = np.linalg.norm(o_complex.positions-o3.positions[0], axis=1)
        cond5 = all_d5 <= props['d5_thresh']
        if np.any(cond5):
            value = True
        return value

def fix_switch_list(flat_switch):
    ndx_ini = int(np.round(max((props['start_ps'] - U.trajectory[0].time)/DT, 0), 0))

    fixed = [i + ndx_ini for i in flat_switch]
    return fixed

def calculate_residence_time(onoff_raw, dists_p, dists_zn):
    onoff = np.zeros((onoff_raw.shape[0]+2, onoff_raw.shape[1]))
    onoff[1:-1] = onoff_raw
    gates = onoff[1:,:] - onoff[:-1,:]
    on_switch, off_switch = [list(np.where(gate==1)[0]+1) for gate in gates.T], [list(np.where(gate==-1)[0]) for gate in gates.T]

    b_res = [[g_sub.residues.ix[i]+1]*len(on) for i, on in enumerate(on_switch)]
    b_res = flatten_list(b_res)

    b_times = [[(b-a+1)*DT for a,b in zip(on, off)] for on, off in zip(on_switch, off_switch)]
    b_times = flatten_list(b_times)

    median_dists = [[np.round(np.median(dists_p[a-1:b,i])/10, 2) for a,b in zip(on,off)] for i,(on,off) in enumerate(zip(on_switch, off_switch))]
    median_dists = flatten_list(median_dists)

    median_zns = [[np.round(np.median(dists_zn[a-1:b,i])/10, 2) for a,b in zip(on,off)] for i,(on,off) in enumerate(zip(on_switch, off_switch))]
    median_zns = flatten_list(median_zns)

    on_switch, off_switch = flatten_list(on_switch), flatten_list(off_switch)
    on_switch, off_switch = fix_switch_list(on_switch), fix_switch_list(off_switch)

    return b_times, median_dists, on_switch, off_switch, b_res, median_zns

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
    if btimes != []:
        hist, bins = np.histogram(btimes, density=True, range=(0,5000), bins=100)
        bins = (bins[1:] + bins[:-1])*0.5
        dx = bins[1] - bins[0]
        first = np.sum(np.multiply(hist*dx, bins))
        second = np.sum(np.multiply(hist*dx, bins**2)) - first**2
        second = np.sqrt(second)
    else:
        first, second = -1, -1
    return first, second

def michaelis():
    result = {}
    ndx_ini = int(np.round(max((props['start_ps'] - U.trajectory[0].time)/DT, 0), 0))
    ndx_fin = int(np.round((props['stop_ps'] - U.trajectory[0].time)/DT,0))
    for ts in U.trajectory[ndx_ini:ndx_fin]:
        if ts.time >= props['start_ps'] and ts.time <= props['stop_ps']:
            print(ts.time, end='\r')
            substrates = []
            for res in sel['SUB'].residues:
                sub_now = Substrate(res)
                substrates.append(sub_now)
            result[ts.time] = substrates
        elif ts.time >= props['stop_ps']:
            break

    n_frames = len(result)
    bound = np.array([[res.bound for res in substrate] for substrate in result.values()])
    precat = np.array([[res.precat for res in substrate] for substrate in result.values()])
    precat_solv = np.array([[res.precat_solv for res in substrate] for substrate in result.values()])
    active = np.array([[res.active for res in substrate] for substrate in result.values()])
    dists_p = np.array([[res.d6 for res in substrate] for substrate in result.values()])
    dists_zn = np.array([[res.dzn for res in substrate] for substrate in result.values()])

    frac_bound = calculate_occupancy(bound)
    frac_precat = calculate_occupancy(precat)
    frac_precat_solv = calculate_occupancy(precat_solv)
    frac_active = calculate_occupancy(active)

    bound_times, bound_dists, bound_on, bound_off, bound_res, bound_zn = calculate_residence_time(bound, dists_p, dists_zn)
    precat_times, precat_dists, precat_on, precat_off, precat_res, precat_zn = calculate_residence_time(precat, dists_p, dists_zn)
    precat_solv_times, precat_solv_dists, precat_solv_on, precat_solv_off, precat_solv_res, precat_solv_zn = calculate_residence_time(precat_solv, dists_p, dists_zn)
    active_times, active_dists, active_on, active_off, active_res, active_zn = calculate_residence_time(active, dists_p, dists_zn)

    bound_moments = statistical_moments(bound_times)
    precat_moments = statistical_moments(precat_times)
    precat_solv_moments = statistical_moments(precat_solv_times)
    active_moments = statistical_moments(active_times)

    return (frac_bound, frac_precat, frac_precat_solv, frac_active),\
    (bound_times, bound_dists, bound_on, bound_off, bound_res, bound_zn, precat_times, precat_dists, precat_on, precat_off, precat_res, precat_zn, precat_solv_times, precat_solv_dists, precat_solv_on, precat_solv_off, precat_solv_res, precat_solv_zn, active_times, active_dists, active_on, active_off, active_res, active_zn),\
    (bound_moments, precat_moments, precat_solv_moments, active_moments)

def write_output(fractions, residence_times, moments):
    states = ["Bound", "Precat", "Precat_solv", "Active"]
    f = open(NAME+"_restimes.sfu", "w")
    #Writes input parameters
    f.write("#Residence time (ps) for bound, precatalytic, precatalytic_solvated, and active complexes, and minimum distance between phosphorus and Au C.O.M.\n")
    f.write("#The reported frames are with respect to the read frames, i.e., it changes with the given dt, tini, tfin.\n")
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    #Writes percental occupations
    f.write("\n#Mean and standard deviation of percentual occupations\n")
    for frac, state in zip(fractions, states):
        f.write("#{:<8.3f} +/- {:>8.3f} ({})\n".format(*frac, state))
    #Writes statistical moments
    f.write("\n#First and second statistical moments of residence times distributions\n")
    for moment, state in zip(moments, states):
        f.write("#{:<8.3f} +/- {:>8.3f} ({})\n".format(*moment, state))
    #Writes residence times
    f.write("\n#{:<7}(ps) {:<8}(nm) {:<8}(fr) {:<8}(fr) {:<7}(res) {:<7}(znm) {:<8}(ps) {:<8}(nm) {:<8}(fr) {:<8}(fr) {:<7}(res) {:<7}(znm) {:<8}(ps) {:<8}(nm) {:<8}(fr) {:<8}(fr) {:<7}(res) {:<7}(znm) {:<8}(ps) {:<8}(nm) {:<8}(fr) {:<8}(fr) {:<7}(res) {:<7}(znm)\n".format(*[x for pair in zip(states, states, states, states, states, states) for x in pair]))
    for pt in zip_longest(*residence_times, fillvalue=''):
        f.write(("{:<12} "*len(residence_times)).format(*pt) + "\n")
    f.close()

fractions, residence_times, moments = michaelis()
write_output(fractions, residence_times, moments)
