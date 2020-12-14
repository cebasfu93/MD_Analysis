import numpy as np
from scipy.spatial.distance import cdist
from MDAnalysis import *
from collections import defaultdict
from tqdm import tqdm

class WaterGroup:
    def __init__(self, res_ix, wat_ixs, time, dist):
        self.res_ix = res_ix
        self.wat_ixs = list(wat_ixs)
        self.time = time
        self.dist = dist


class ExchangeEvent:
    def __init__(self, wg):
        self.rtarget = wg.res_ix
        self.xref = wg.wat_ixs
        self.time = [wg.time]
        self.dist = [wg.dist]
        self.alive = False

    def extend_event(self, wg):
        self.xref += wg.wat_ixs
        self.time.append(wg.time)
        self.dist.append(wg.dist)

    def compress_event(self, U, props, DT, func=np.mean):
        self.anchor_dist = func(self.dist)/10 #A to nm
        self.tmin = min(self.time)
        self.tmax = max(self.time) + DT
        self.duration = self.tmax - self.tmin
        self.ex_rate = len(set(self.xref))/self.duration
        if props['stop_ps'] - DT in self.time:
            self.alive = True


def search_exchange_events(U, props, wgs, DT):
    events = {}
    counters = defaultdict(int)

    print("Searching events")
    for w, wg in tqdm(enumerate(wgs), total=len(wgs)):
        key = (wg.res_ix, counters[wg.res_ix])
        if key not in events.keys():
            events[key] = ExchangeEvent(wg)
        else:
            if wg.time-DT in events[key].time:
                events[key].extend_event(wg)
            else:
                counters[wg.res_ix] += 1
                key = (wg.res_ix, counters[wg.res_ix])
                events[key] = ExchangeEvent(wg)

    events = list(events.values())
    print("Compressing events")
    for e, event in tqdm(enumerate(events), total=len(events)):
        event.compress_event(U, props, DT)
    return events


def water_exchange(U, props, sel):
    g_anchor = sel[props['anchor']]
    g_ref = sel[props['ref']]
    n_read = int((props['stop_ps'] - props['start_ps'])//U.trajectory[0].dt)
    all_wgs = defaultdict(list)
    for target in props['targets']:
        res_target = sel[target].residues
        g_residues = [res.atoms for res in res_target]
        res_Hs = [res.intersection(sel[target]) for res in g_residues]
        ref_Hs = [U.select_atoms("(around {} group AN) and group REF".format(props['d_max']), REF=g_ref, AN=res, updating=True) for res in res_Hs]
        bound = np.zeros((n_read+2, len(res_Hs)), dtype='int') #+2 to start and finish with False
        for t, ts in tqdm(enumerate(U.trajectory,1), total=n_read):
            if ts.time >= props['stop_ps']:
                break
            elif ts.time >= props['start_ps']:
                for r, res in enumerate(res_target):
                    anchor_dist = np.linalg.norm(res.atoms.center_of_mass() - g_anchor.center_of_mass())
                    if anchor_dist <= props['d_tumbling']:
                        wg = WaterGroup(res.ix, ref_Hs[r].ix, ts.time, anchor_dist)
                        all_wgs[target].append(wg)
    return all_wgs


def write_exchange_events(events, props, sel, name):
    f = open(name + "_wexchange.sfu", 'w')

    f.write("#Water protons exhange rate (Hs ps-1) of targets within a tumbling distance of the anchor\n")
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))

    f.write("#*** means that the binding event reached the end of the simulation\n")
    f.write("#Resid target \t Starting time (ps) \t Ending time (ps) \t Duration (ps) \t Exchange rate (ps-1) \t Mean target-anchor COM distance (nm)\n")

    for targets in props['targets']:
        f.write("#TARGET GROUP: {}\n".format(targets))
        event = events[targets]
        for ev in event:
            f.write("{:<5} {:<10.0f} {:<10.0f} {:<10.0f} {:<10.2f} {:<10.2f}".format(ev.rtarget, ev.tmin, ev.tmax, ev.duration, ev.ex_rate, ev.anchor_dist))
            if ev.alive:
                f.write(" ***")
            f.write("\n")

    f.close()


def pipeline_water_exchange(U, props, sel, name):
    DT = U.trajectory[0].dt
    trajs = water_exchange(U, props, sel)
    events = {key : search_exchange_events(U, props, val, DT) for key, val in trajs.items()}
    write_exchange_events(events, props, sel, name)
