import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from MDAnalysis import *
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt


def shell_exchange(U, props, sel):
    g_ref = sel[props['ref']]
    g_target = sel[props['target']]
    # interface 0 is between shell 0 and 1 (i.e. between bin pivot 1 and 2)
    interfaces = {i: 0 for i in range(props['bins'] - 1)}
    bins = np.linspace(*props['r_range'], num=props['bins'] + 1)
    db = bins[1] - bins[0]
    DT = U.trajectory[0].dt
    n_read = int(props['stop_ps'] - props['start_ps']) / DT + 1
    all_locations = []
    for ts in tqdm(U.trajectory, total=n_read):
        if ts.time > props['stop_ps']:
            break
        elif ts.time > props['start_ps']:
            dx = g_target.positions - g_ref.center_of_mass()
            norms = np.linalg.norm(dx, axis=1)
            locations = np.clip((norms // db).astype('int'), 0, props['bins'], dtype='int')
            all_locations += list(prev_locations)
            # The less sign implies that we look at events where water enters the shell, not when they leave
            mask_enter = np.less(locations, prev_locations)
            # We track the locations from which the water just came
            doping = prev_locations[mask_enter]
            shells, counts = np.unique(doping, return_counts=True)
            for shell, count in zip(shells, counts):
                if shell < props['bins'] - 1:
                    interfaces[shell] += count
            prev_locations = locations
        elif ts.time == props['start_ps']:
            dx = g_target.positions - g_ref.center_of_mass()
            norms = np.linalg.norm(dx, axis=1)
            prev_locations = np.clip((norms // db).astype('int'),
                                     0, props['bins'], dtype='int')
    # otherwise integer locations dont fall exactly at the bin pivots
    all_locations = np.array(all_locations) + 0.5
    hist, _ = np.histogram(all_locations, bins=np.linspace(
        0, props['bins'], props['bins'] + 1), density=False)

    for key, h in zip(interfaces.keys(), hist[:-2]):
        n_T = np.sum(all_locations < props['bins'] - 1)
        n_shell = hist[key]
        # 100 for A-2 to nm-2
        norm = 3 * n_T * db / (4 * np.pi * ((props['r_range'][1])**3) * n_shell * n_read) * 100
        norm = 0 if norm == np.inf else norm
        interfaces[key] *= norm

    fig = plt.figure()
    ax = plt.axes()
    ax.set_xticks(np.linspace(1, props['bins'] + 1, 5))
    ax.set_xticklabels(np.linspace(*props['r_range'], num=5))
    ax.plot(list(interfaces.keys())[:-1], list(interfaces.values())[:-1])
    plt.show()

    return bins, interfaces


def write_water_shell_exchange(bins, trajs, props, name):
    df = pd.DataFrame(zip(trajs.keys(), trajs.values()), columns=['shell', 'rate']).sort_values('shell')
    f = open(name + "_wshellexhange.sfu", 'w')
    f.write("#Rate at which water molecules leave a particular shell\n")
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    f.write("#Distance to ref (nm) \t Residence time (ps)\n")
    f.write("{:<10.2f} {:>10.3f}\n".format(0.0, 0.0))
    for i in range(props['bins'] - 2):
        f.write("{:<10.2f} {:>10.3f}\n".format(bins[i + 1], trajs.get(i, 0.0)))
    f.close()


def pipeline_water_shell_exchange(U, props, sel, name):
    DT = U.trajectory[0].dt
    bins, trajs = shell_exchange(U, props, sel)
    write_water_shell_exchange(bins, trajs, props, name)
