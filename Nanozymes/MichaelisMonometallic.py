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
'stop_ps'   : 10000000,# 567
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

        BOUND, g_zn_complex = self.is_bound(o4, o5)
        if BOUND:
            PRE1 = self.is_pre1(o3, g_zn_complex)
            if PRE1:
                PRE2, SOLV2, ACT2, PRE3, SOLV3, ACT3 = False, False, False, False, False, False
                SOLV1, g_o_complex = self.is_solv12(g_zn_complex)
                if SOLV1:
                    ACT1 = self.is_act123(g_o_complex, o3)
                else:
                    ACT1 = False
            else:
                SOLV1, ACT1 = False, False
                PRE2, g_zn2_complex = self.is_pre2(o3, g_zn_complex)
                if PRE2:
                    PRE3, SOLV3, ACT3 = False, False, False
                    SOLV2, g_o_complex = self.is_solv12(g_zn2_complex)
                    if SOLV2:
                        ACT2 = self.is_act123(g_o_complex, o3)
                    else:
                        ACT2 = False
                else:
                    SOLV2, ACT2 = False, False
                    PRE3, g_zn_both = self.is_pre3(o3, g_zn_complex)
                    if PRE3:
                        SOLV3, g_o_complex = self.is_solv3(g_zn_both)
                        if SOLV3:
                            ACT3 = self.is_act123(g_o_complex, o3)
                        else:
                            ACT3 = False
                    else:
                        SOLV3, ACT3 = False, False
        else:
            PRE1, SOLV1, ACT1, PRE2, SOLV2, ACT2, PRE3, SOLV3, ACT3 = False, False, False, False, False, False, False, False, False

        self.bound = BOUND
        self.pre1, self.solv1, self.act1 = PRE1, SOLV1, ACT1
        self.pre2, self.solv2, self.act2 = PRE2, SOLV2, ACT2
        self.pre3, self.solv3, self.act3 = PRE3, SOLV3, ACT3

        self.d6 = np.linalg.norm(ph.positions[0] - g_au.center_of_mass())

    def is_bound(self, o4, o5):
        value = False
        all_d1 = np.linalg.norm(g_zn.positions - o5.positions[0], axis=1)
        cond_d1 = all_d1 <= props['d1_thresh']
        g_zn_d1 = U.atoms[g_zn[np.where(cond_d1)].ix]

        all_d2 = np.linalg.norm(g_zn.positions - o4.positions[0], axis=1)
        cond_d2 = all_d2 <= props['d1_thresh']
        g_zn_d2 = U.atoms[g_zn[np.where(cond_d2)].ix]

        if len(g_zn_d1.union(g_zn_d2)) == 1:
            value = True

        return value, g_zn_d1.union(g_zn_d2)

    def is_pre1(self, o3, zn_complex):
        value = False
        all_d3 = np.linalg.norm(zn_complex.positions[0] - o3.positions[0])
        if all_d3 <= props['d3_thresh']:
            value = True
        return value

    def is_solv12(self, zn_complex):
        value = False
        all_d4 = np.linalg.norm(g_ow.positions - zn_complex.positions[0], axis=1)
        cond4 = all_d4 <= props['d4_thresh']
        o_complex = g_ow[np.where(cond4)[0]]
        if np.any(cond4):
            value = True
        return value, o_complex

    def is_act123(self, o_complex, o3):
        value = False
        all_d5 = np.linalg.norm(o_complex.positions-o3.positions[0], axis=1)
        cond5 = all_d5 <= props['d5_thresh']
        if np.any(cond5):
            value = True
        return value

    def is_pre2(self, o3, zn_complex):
        value = False
        g_zn_not1 = g_zn.difference(zn_complex)
        all_d3 = np.linalg.norm(g_zn_not1.positions - o3.positions[0], axis=1)
        cond3 = all_d3 <= props['d3_thresh']
        g_zn2 = g_zn_not1[np.where(cond3)[0]]
        if np.any(cond3):
            value = True
        return value, g_zn2

    def is_pre3(self, o3, g_zn_complex):
        value = False
        g_zn_not1 = g_zn.difference(g_zn_complex)
        all_d3_not1 = np.linalg.norm(g_zn_not1.positions - o3.positions[0], axis=1)
        all_d3_1 = np.linalg.norm(g_zn_complex.positions[0] - o3.positions[0])
        cond3_not1 = all_d3_not1 <= props['d3_thresh']
        cond3_1 = all_d3_1 <= props['d3_thresh']

        g_zn2 = g_zn_not1[np.where(cond3_not1)[0]]
        if cond3_1 and np.any(cond3_not1):
            value = True
        return value, g_zn_complex.union(g_zn2)

    def is_solv3(self, zn_both):
        value = False
        all_d4 = cdist(g_ow.positions, zn_both.positions)
        cond4 = all_d4 <= props['d4_thresh']
        o_complex = g_ow[np.where(cond4)[0]]
        if np.any(cond4):
            value = True
        return value, o_complex

def fix_switch_list(flat_switch):
    ndx_ini = int(np.round(max((props['start_ps'] - U.trajectory[0].time)/DT, 0), 0))

    fixed = [i + ndx_ini for i in flat_switch]
    return fixed

def calculate_residence_time(onoff_raw, dists_p):
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

    on_switch, off_switch = flatten_list(on_switch), flatten_list(off_switch)
    on_switch, off_switch = fix_switch_list(on_switch), fix_switch_list(off_switch)

    return b_times, median_dists, on_switch, off_switch, b_res

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
    estado = {}
    estado['bound'] = np.array([[res.bound for res in substrate] for substrate in result.values()])
    estado['pre1'] = np.array([[res.pre1 for res in substrate] for substrate in result.values()])
    estado['solv1'] = np.array([[res.solv1 for res in substrate] for substrate in result.values()])
    estado['act1'] = np.array([[res.act1 for res in substrate] for substrate in result.values()])
    estado['pre2'] = np.array([[res.pre2 for res in substrate] for substrate in result.values()])
    estado['solv2'] = np.array([[res.solv2 for res in substrate] for substrate in result.values()])
    estado['act2'] = np.array([[res.act2 for res in substrate] for substrate in result.values()])
    estado['pre3'] = np.array([[res.pre3 for res in substrate] for substrate in result.values()])
    estado['solv3'] = np.array([[res.solv3 for res in substrate] for substrate in result.values()])
    estado['act3'] = np.array([[res.act3 for res in substrate] for substrate in result.values()])

    dists_p = np.array([[res.d6 for res in substrate] for substrate in result.values()])

    fracs = {key : calculate_occupancy(val) for key, val in estado.items()}

    restimes = {key : calculate_residence_time(val, dists_p) for key, val in estado.items()}
    #bound_times, bound_dists, bound_on, bound_off, bound_res = calculate_residence_time(bound, dists_p)

    moments = {key : statistical_moments(val[0]) for key, val in restimes.items()}

    return fracs, restimes, moments

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
        f.write("#{:<7}(ps) {:<8}(nm) {:<8}(fr) {:<8}(fr) {:<8}(id)\n".format("Restime", "DistP", "Start", "End", "Resid"))
        for pt in zip(*val):
            f.write(("{:<12} "*len(pt)).format(*pt) + "\n")
    f.close()

fractions, residence_times, moments = michaelis()
write_output(fractions, residence_times, moments)
