import numpy as np
from scipy.spatial.distance import cdist
from MDAnalysis import *
from collections import defaultdict
from tqdm import tqdm

class Pair:
    """
    Object that stores pairs of atoms in contact (when Hs are distinguishable).
    """
    def __init__(self, ix_ref, ix_target, ndx_gref, ndx_gtarget, res_ref, res_target, time, dist):
        assert len(ndx_gref) == 1
        assert len(ndx_gtarget) == 1
        self.ix_ref = ix_ref             #atomid of ref
        self.ix_target = ix_target      #atomid of target
        self.aref = ndx_gref[0]         #chemicalid of ref
        self.atarget = ndx_gtarget[0]   #chemicalid of target
        self.rref = res_ref             #resid of ref
        self.rtarget = res_target       #resid of target
        self.time = time                #time when the pair is registered
        self.dist = dist                #distance between target and anchor

class ResPair:
    """
    Object that stores residues that are in contact (when Hs are not distinguishable).
    """
    def __init__(self, res_target, time, dist):
        self.rtarget = res_target
        self.time = time
        self.dist = dist

class STDEvent:
    """
    Object that stores a (Water) STD event properties (when Hs are distinguishable).
    """
    def __init__(self, pair):
        self.aref = pair.aref
        self.atarget = pair.atarget
        self.rref = pair.rref
        self.rtarget = pair.rtarget
        self.time = [pair.time]
        self.dist = [pair.dist]
        self.alive = False

    def extend_event(self, pair):
        """
        Catenates two Pair objects belonging to a single event.
        """
        self.time.append(pair.time)
        self.dist.append(pair.dist)

    def compress_event(self, U, props, DT, func=np.mean):
        """
        Calculates summary properties of the event.
        That is, those that will be ultimately printed out.
        The distances to the anchor are combined with the function func.
        """
        self.anchor_dist = func(self.dist)/10 #A to nm
        self.tmin = min(self.time)
        self.tmax = max(self.time) + DT
        self.duration = self.tmax - self.tmin
        if props['stop_ps'] - DT in self.time:
            self.alive = True

class ResSTDEvent:
    """
    Object that stores a (Water) STD event properties (when Hs are not distinguishable).
    """
    def __init__(self, respair):
        self.rtarget = respair.rtarget
        self.time = [respair.time]
        self.dist = [respair.dist]
        self.alive = False

    def extend_event(self, respair):
        """
        Catenates two ResPair objects belonging to a single event.
        """
        self.time.append(respair.time)
        self.dist.append(respair.dist)

    def compress_event(self, U, props, DT, func=np.mean):
        """
        Calculates summary properties of the event.
        That is, those that will be ultimately printed out.
        The distances to the anchor are combined with the function func.
        """
        self.anchor_dist = func(self.dist)/10 #A to nm
        self.tmin = min(self.time)
        self.tmax = max(self.time)
        self.duration = self.tmax - self.tmin
        if props['stop_ps'] - DT in self.time:
            self.alive = True

def search_STDevents(U, props, pairs, DT):
    """
    Combines the list (pairs) of objects Pair that belong to the same binding event (when Hs are distinguishable).
    """
    events = {}
    counters = defaultdict(int)

    print("Searching events")
    for p, pair in tqdm(enumerate(pairs), total=len(pairs)):
        prefix = (pair.ix_ref, pair.ix_target, pair.aref, pair.atarget, pair.rref, pair.rtarget)
        key = (*prefix, counters[prefix])

        if key not in events.keys():
            events[key] = STDEvent(pair)
        else:
            if pair.time-DT in events[key].time:
                events[key].extend_event(pair)
            else:
                counters[prefix] += 1
                key = (*prefix, counters[prefix])
                events[key] = STDEvent(pair)

    events = list(events.values())
    print("Compressing events")
    for e, event in tqdm(enumerate(events), total=len(events)):
        event.compress_event(U, props, DT)
    return events

def search_resSTDevents(U, props, respairs, DT):
    """
    Combines the list (pairs) of objects Pair that belong to the same binding event (when Hs are not distinguishable).
    """
    events = {}
    counters = defaultdict(int)
    print("Searching events")
    for p, respair in tqdm(enumerate(respairs), total=len(respairs)):
        key = (respair.rtarget, counters[respair.rtarget])
        if key not in events.keys():
            events[key] = ResSTDEvent(respair)
        else:
            if respair.time-DT in events[key].time:
                events[key].extend_event(respair)
            else:
                counters[respair.rtarget] += 1
                key = (respair.rtarget, counters[respair.rtarget])
                events[key] = ResSTDEvent(respair)
    events = list(events.values())
    print("Compressing events")
    for e, event in tqdm(enumerate(events), total=len(events)):
        event.compress_event(U, props, DT)
    return events

def binding_time_STD(U, props, sel):
    """
    Makes dictionary with lists of objects type Pair.
    The output pairs are the H-H contacts that account for normal STD events (when Hs are distinguishable).
    """
    all_pairs = {}
    g_anchor = sel[props['anchor']]
    n_read = (props['stop_ps'] - props['start_ps'])//U.trajectory[0].dt + 1

    for targets in props['targets']:
        print("Reading target: {}".format(targets))
        g_all_refs, g_all_targets = AtomGroup([],U), AtomGroup([],U)
        for g in sel[props['ref']]:
            g_all_refs = g_all_refs.union(g)
        for g in sel[targets]:
            g_all_targets = g_all_targets.union(g)
        targets_list = sel[targets]
        refs_list = sel[props['ref']]

        pairs = []
        for ts in tqdm(U.trajectory, total=n_read):
            if ts.time >= props['stop_ps']:
                break
            elif ts.time >= props['start_ps']:
                anchor_dist = {res.ix : np.linalg.norm(res.atoms.center_of_mass()-g_anchor.center_of_mass()) for res in g_all_targets.residues}
                dists = cdist(g_all_refs.positions, g_all_targets.positions)

                ndx_close = np.where(dists<=props['d_max'])
                for i, j in zip(*ndx_close):
                    ix_ref, ix_target = g_all_refs[i].ix, g_all_targets[j].ix
                    ndx_gref = [i for i, g in enumerate(refs_list) if ix_ref in g.ix]
                    ndx_gtarget = [i for i, g in enumerate(targets_list) if ix_target in g.ix]
                    res_ref = U.atoms[ix_ref].resid
                    res_target = U.atoms[ix_target].resid
                    pair = Pair(ix_ref, ix_target, ndx_gref, ndx_gtarget, res_ref, res_target, ts.time, anchor_dist[res_target])
                    pairs.append(pair)

        all_pairs[targets] = pairs

    return all_pairs

def binding_time_resSTD(U, props, sel):
    """
    Makes dictionary with lists of objects type ResPair.
    The output pairs are the H-H contacts that account for normal STD events (when Hs are not distinguishable).
    """
    all_respairs = defaultdict(list)
    g_anchor = sel[props['anchor']]
    n_read = int((props['stop_ps'] - props['start_ps'])//U.trajectory[0].dt)
    ref_Hs = sel[props['ref']]

    for target in props['targets']:
        tar_res = sel[target].residues
        tar_Hs = [res.atoms.intersection(sel[target]) for res in tar_res]

        for t, ts in tqdm(enumerate(U.trajectory,1), total=n_read):
            if ts.time >= props['stop_ps']:
                break
            elif ts.time >= props['start_ps']:
                for r, res in enumerate(tar_res):
                    dists = cdist(tar_Hs[r].positions, ref_Hs.positions)
                    if np.any(dists<=props['d_max']):
                        anchor_dist = np.linalg.norm(res.atoms.center_of_mass() - g_anchor.center_of_mass())
                        respair = ResPair(res.ix, ts.time, anchor_dist)
                        all_respairs[target].append(respair)

    return all_respairs

def binding_time_WSTD(U, props, sel):
    """
    Makes dictionary with lists of objects type Pair.
    The output pairs are the H-H contacts that account for Water STD events (when Hs are distinguishable).
    """
    all_pairs = {}
    g_anchor = sel[props['anchor']]
    n_read = (props['stop_ps'] - props['start_ps'])//U.trajectory[0].dt + 1

    for targets in props['targets']:
        print("Reading target: {}".format(targets))
        g_all_refs, g_all_targets = AtomGroup([],U), AtomGroup([],U)
        for g in sel[props['ref']]:
            g_all_refs = g_all_refs.union(g)
        for g in sel[targets]:
            g_all_targets = g_all_targets.union(g)
        targets_list = sel[targets]
        refs_list = sel[props['ref']]

        pairs = []
        for ts in tqdm(U.trajectory, total=n_read):
            if ts.time >= props['stop_ps']:
                break
            elif ts.time >= props['start_ps']:
                anchor_dist = {res.ix : np.linalg.norm(res.atoms.center_of_mass()-g_anchor.center_of_mass()) for res in g_all_targets.residues}
                dists = cdist(g_all_refs.positions, g_all_targets.positions)

                ndx_close = np.where(dists<=props['d_max'])
                for i, j in zip(*ndx_close):
                    target_dist = anchor_dist[g_all_targets[j].residue.ix]
                    if target_dist <= props['d_tumbling']:
                        ix_ref, ix_target = g_all_refs[i].ix, g_all_targets[j].ix
                        ndx_gref = [i for i, g in enumerate(refs_list) if ix_ref in g.ix]
                        ndx_gtarget = [i for i, g in enumerate(targets_list) if ix_target in g.ix]
                        res_ref = U.atoms[ix_ref].resid
                        res_target = U.atoms[ix_target].resid
                        pair = Pair(ix_ref, ix_target, ndx_gref, ndx_gtarget, res_ref, res_target, ts.time, anchor_dist[res_target])
                        pairs.append(pair)

        all_pairs[targets] = pairs

    return all_pairs

def binding_time_resWSTD(U, props, sel):
    """
    Makes dictionary with lists of objects type Pair.
    The output pairs are the H-H contacts that account for Water STD events (when Hs are not distinguishable).
    """
    all_respairs = defaultdict(list)
    g_anchor = sel[props['anchor']]
    n_read = int((props['stop_ps'] - props['start_ps'])//U.trajectory[0].dt)
    all_ref_Hs = sel[props['ref']]

    for target in props['targets']:
        tar_res = sel[target].residues
        tar_Hs = [res.atoms.intersection(sel[target]) for res in tar_res]
        ref_Hs = [U.select_atoms("(around {} group HTAR) and group HREF".format(props['d_max']), HREF=all_ref_Hs, HTAR=tar_H, updating=True) for tar_H in tar_Hs]

        for t, ts in tqdm(enumerate(U.trajectory,1), total=n_read):
            if ts.time >= props['stop_ps']:
                break
            elif ts.time >= props['start_ps']:
                for r, res in enumerate(tar_res):
                    anchor_dist = np.linalg.norm(res.atoms.center_of_mass() - g_anchor.center_of_mass())
                    if anchor_dist <= props['d_tumbling'] and len(ref_Hs[r]) > 0:
                        respair = ResPair(res.ix, ts.time, anchor_dist)
                        all_respairs[target].append(respair)

    return all_respairs

def write_events_STD(events, props, sel, name, identical_ref, identical_target):
    """
    Writes a file describing all the STD events (when Hs are distinguishable).
    The output includes atom indices, residue indices, and chemical position of the reference and target groups
    as well as initial time, final time, duration, and distance to the anchor for each event.
    """
    f = open(name + "_btimes_STD.sfu", 'w')

    f.write("#Binding residence time (ps) from (nonpolar) H-H contacts\n")
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))

    f.write("#Reference groups (aref)\n")
    for t, txt in enumerate(identical_ref):
        f.write("\t# {} --> {:<30}\n".format(t, txt))
    for targets in props['targets']:
        f.write("#Target groups (atarget) for {}\n".format(targets))
        for t, txt in enumerate(identical_target[targets]):
            f.write("\t# {} --> {:<30}\n".format(t, txt))

    f.write("#*** means that the binding event reached the end of the simulation\n")
    f.write("#aref \t atarget \t Resid ref \t Resid target \t Starting time (ps) \t Ending time (ps) \t Duration (ps) \t Mean target-ref COM distance (nm)\n")

    for targets in props['targets']:
        f.write("#TARGET GROUP: {}\n".format(targets))
        event = events[targets]
        for ev in event:
            f.write("{:<5} {:<5} {:<5} {:<5} {:<10.0f} {:<10.0f} {:<10.0f} {:<10.2f}".format(ev.aref, ev.atarget, ev.rref, ev.rtarget, ev.tmin, ev.tmax, ev.duration, ev.anchor_dist))
            if ev.alive:
                f.write(" ***")
            f.write("\n")

    f.close()

def write_events_resSTD(events, props, sel, name):
    """
    Writes a file describing all the STD events (when Hs are not distinguishable).
    The output includes residue index of the target groups
    as well as initial time, final time, duration, and distance to the anchor for each event.
    """
    f = open(name + "_btimes_resSTD.sfu", 'w')

    f.write("#Binding residence time (ps) from (nonpolar) H-H contacts (the Hs of the monolayer and an analyte are all considered identical)\n")
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))

    f.write("#*** means that the binding event reached the end of the simulation\n")
    f.write("#Resid target \t Starting time (ps) \t Ending time (ps) \t Duration (ps) \t Mean target-ref COM distance (nm)\n")

    for target in props['targets']:
        f.write("#TARGET GROUP: {}\n".format(target))
        event = events[target]
        for ev in event:
            f.write("{:<5} {:<10.0f} {:<10.0f} {:<10.0f} {:<10.2f}".format(ev.rtarget, ev.tmin, ev.tmax, ev.duration, ev.anchor_dist))
            if ev.alive:
                f.write(" ***")
            f.write("\n")

    f.close()

def write_events_WSTD(events, props, sel, name, identical_ref, identical_target):
    """
    Writes a file describing all the Water STD events (when Hs are distinguishable).
    The output includes atom indices, residue indices, and chemical position of the reference and target groups
    as well as initial time, final time, duration, and distance to the anchor for each event.
    """
    f = open(name + "_btimes_WSTD.sfu", 'w')

    f.write("#Binding residence time (ps) from (nonpolar) H-H contacts\n")
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))

    f.write("#Reference groups (aref)\n")
    for t, txt in enumerate(identical_ref):
        f.write("\t# {} --> {:<30}\n".format(t, txt))
    for targets in props['targets']:
        f.write("#Target groups (atarget) for {}\n".format(targets))
        for t, txt in enumerate(identical_target[targets]):
            f.write("\t# {} --> {:<30}\n".format(t, txt))

    f.write("#*** means that the binding event reached the end of the simulation\n")
    f.write("#aref \t atarget \t Resid ref \t Resid target \t Starting time (ps) \t Ending time (ps) \t Duration (ps) \t Mean target-ref COM distance (nm)\n")

    for targets in props['targets']:
        f.write("#TARGET GROUP: {}\n".format(targets))
        event = events[targets]
        for ev in event:
            f.write("{:<5} {:<5} {:<5} {:<5} {:<10.0f} {:<10.0f} {:<10.0f} {:<10.2f}".format(ev.aref, ev.atarget, ev.rref, ev.rtarget, ev.tmin, ev.tmax, ev.duration, ev.anchor_dist))
            if ev.alive:
                f.write(" ***")
            f.write("\n")

    f.close()

def write_events_resWSTD(events, props, sel, name):
    """
    Writes a file describing all the Water STD events (when Hs are not distinguishable).
    The output includes residue index of the target groups
    as well as initial time, final time, duration, and distance to the anchor for each event.
    """
    f = open(name + "_btimes_resWSTD.sfu", 'w')

    f.write("#Binding residence time (ps) from (nonpolar) H-H contacts (the Hs of the tumbling water and an analyte are all considered identical)\n")
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))

    f.write("#*** means that the binding event reached the end of the simulation\n")
    f.write("#Resid target \t Starting time (ps) \t Ending time (ps) \t Duration (ps) \t Mean target-ref COM distance (nm)\n")

    for target in props['targets']:
        f.write("#TARGET GROUP: {}\n".format(target))
        event = events[target]
        for ev in event:
            f.write("{:<5} {:<10.0f} {:<10.0f} {:<10.0f} {:<10.2f}".format(ev.rtarget, ev.tmin, ev.tmax, ev.duration, ev.anchor_dist))
            if ev.alive:
                f.write(" ***")
            f.write("\n")

    f.close()

def pipeline_STD(U, props, sel, name, identical_ref, identical_target):
    """
    Pipeline for calculating STD events (i.e. monolayer-analyte) considering each H-H contact as a different pair
    """
    DT = U.trajectory[0].dt
    trajs = binding_time_STD(U, props, sel)
    events = {key : search_STDevents(U, props, val, DT) for key, val in trajs.items()}
    write_events_STD(events, props, sel, name, identical_ref, identical_target)

def pipeline_resSTD(U, props, sel, name):
    """
    Pipeline for calculating STD events (i.e. monolayer-analyte) considering any hydrogen from the monolayer and the analyte
    """
    DT = U.trajectory[0].dt
    trajs = binding_time_resSTD(U, props, sel)
    events = {key : search_resSTDevents(U, props, val, DT) for key, val in trajs.items()}
    write_events_resSTD(events, props, sel, name)

def pipeline_WSTD(U, props, sel, name, identical_ref, identical_target):
    """
    Pipeline for calculating Water STD events (i.e. NP bound water-analyte) considering each H-H contact as a different pair
    """
    DT = U.trajectory[0].dt
    trajs = binding_time_WSTD(U, props, sel)
    events = {key : search_STDevents(U, props, val, DT) for key, val in trajs.items()}
    write_events_WSTD(events, props, sel, name, identical_ref, identical_target)

def pipeline_resWSTD(U, props, sel, name):
    """
    Pipeline for calculating Water STD events (i.e. NP bound water-analyte) considering any hydrogen from the monolayer and the water.
    This will be practically the same as considering a binding event when an analyte is within d_tumbling from the anchor COM
    """
    DT = U.trajectory[0].dt
    trajs = binding_time_resWSTD(U, props, sel)
    events = {key : search_resSTDevents(U, props, val, DT) for key, val in trajs.items()}
    write_events_resWSTD(events, props, sel, name)
