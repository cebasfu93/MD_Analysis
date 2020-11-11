"""
For every binding event, it reports the minimum analyte-gold inter-COM distance and the binding residence time.
"""
XTC = "NP22sp-53_PRO1-10_FIX.xtc"
TPR = "NP22sp-53_PRO1.tpr"
NAME = XTC[:-8]

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from MDAnalysis import *

U = Universe(TPR, XTC)
print(len(U.trajectory))
DT = U.trajectory[0].dt
sel = {
"all_gold" : U.select_atoms("name AU AUS AUL"),
"mono_H"    : U.select_atoms("resname L22 and name H* and not name H18"),
"SER_H"     : U.select_atoms("resname SER and name H* and not name H5 H6 H7 H9 H13"),
"PHE_H"     : U.select_atoms("resname PHE and name H* and not name H9 H10 H11"),
"SOL"       : U.select_atoms("resname SOL")
}

props_bind_time = {
'anchor'    : 'all_gold', #the script reports the minimum distance respect to this group
'ref'       : "mono_H",
'targets'   : ["SER_H", "PHE_H"],
'solvent'   : "SOL",
'start_ps'  : 0,
'stop_ps'   : 1000000,
'd_max'     : 4, #A, threshold distance for magnetization transfer
}

flatten_list = lambda l: [item for sublist in l for item in sublist]
def flatten_group_list(l):
    res = []
    for sublist in l:
        group = sublist[0]
        for i, item in enumerate(sublist[1:]):
            group = group.union(item)
        res.append(group)
    return res

class BindingEvent:
    def __init__(self, frameini, framefin, Residue, target, props):
        g_sol = sel[props['solvent']]
        g_anchor = sel[props['anchor']]
        g_res_hs = Residue.atoms.intersection(sel[target])

        self.a = frameini
        self.b = framefin
        self.resid = Residue.resid+1
        self.duration = (self.b - self.a)*DT
        meddist, self.waters = [], AtomGroup([], U)
        for ts in U.trajectory[self.a:self.b]:
            meddist.append(np.linalg.norm(Residue.atoms.center_of_mass() - g_anchor.center_of_mass()))
            g_wat_now = U.select_atoms("(around {} group AN) and group SOL".format(props['d_max']), AN=g_res_hs, SOL=g_sol, updating=True)
            self.waters = self.waters.union(g_wat_now)
        self.meddist = np.median(meddist)
        self.v_wat = self.waters.n_residues/self.duration

def bind_time(props):
    g_anchor = sel[props['anchor']]
    g_ref = sel[props['ref']]
    all_events = {}

    for target in props['targets']:
        res_target = sel[target].residues
        g_residues = [res.atoms for res in sel[target].residues]
        res_Hs = [res.intersection(sel[target]) for res in g_residues]
        bound = np.zeros((len(U.trajectory)+2, len(res_Hs)), dtype='int') #+2 to start and finish with False
        for t, ts in enumerate(U.trajectory,1):
            if ts.time >= props['stop_ps']:
                break
            elif ts.time >= props['start_ps']:
                print(ts.time, end='\r')
                dists = [cdist(g_ref.positions, res.positions) for res in res_Hs]
                close = np.array([np.any(dist<= props['d_max']) for dist in dists])
                bound[t,close] = 1

        gates = bound[1:,:] - bound[:-1,:]
        on_switch, off_switch = [list(np.where(gate==1)[0]) for gate in gates.T], [list(np.where(gate==-1)[0]) for gate in gates.T]
        events = []
        n_events = len(flatten_list(on_switch))
        ev = 1
        print('')
        for r, res in enumerate(res_target):
            for on, off in zip(on_switch[r], off_switch[r]):
                print("{:d}/{:d}".format(ev, n_events), end="\r")
                event = BindingEvent(on, off, res, target=target, props=props)
                events.append(event)
                ev += 1
        all_events[target] = events
    return all_events

def write_bind_time(props, all_events):
    f = open(NAME + "_btimes.sfu", 'w')
    values = []
    f.write("#Binding residence time (ps) from (nonpolar) H-H contacts\n")
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    f.write("#*** means that the binding event reached the end of the simulation\n")
    f.write("#Binding residence time (ps) \t Median inter-COM distance between anchor and target during the binding event (nm) \t Number of unique waters around target per unit time (ps-1) \t VMD resid \n")
    for target in props['targets']:
        f.write("#TARGET GROUP: {}\n".format(target))
        events = all_events[target]
        for event in events:
            f.write("{:<10.1f}  {:>10.3f}  {:>10.3f}  {:>7d}".format(event.duration, event.meddist, event.v_wat, event.resid))
            if event.b*DT == props['stop_ps']:
                f.write(" ***")
            f.write("\n")

    f.close()

all_events = bind_time(props_bind_time)
write_bind_time(props_bind_time, all_events)
