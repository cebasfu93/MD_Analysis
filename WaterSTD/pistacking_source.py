import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import matplotlib.pyplot as plt


def angle_between_vectors(v1, v2):
    """
    Returns angle (deg) between v1 and v2 [0 - 180]
    """
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos, -1, 1))
    return angle * 180 / np.pi


def angle_between_planes(ag1, ag2):
    """
    Returns angle (deg) between the molecular plane of two rings [0 - 180]
    ag1 and ag2 are atom groups (e.g. carbon atoms in an aromatic cycle)
    The molecular plane is the one containing two vectors:
    i) From the centroid of the atoms to the first atom in the AtomGroup
    ii) from the centroid of the atoms to the third atom in the AtomGroup
    """
    center1, center2 = ag1.center_of_geometry(), ag2.center_of_geometry()
    v1a = ag1.atoms[0].position - center1
    v1b = ag1.atoms[2].position - center1
    norm1 = np.cross(v1a, v1b)
    v2a = ag2.atoms[0].position - center2
    v2b = ag2.atoms[2].position - center2
    norm2 = np.cross(v2a, v2b)
    angle = angle_between_vectors(norm1, norm2)
    angle = 180 - angle if angle > 90 else angle  # This angle is cyclic. 91 deg is the same as 89 deg
    return angle


def angle_between_plane_vector(ag1, ag2):
    """
    Returns the angle (deg) between the molecular plane of one ring and the vector connecting the centroid off two AtomGroups
    ag1 and ag2 are atom groups (e.g. carbon atoms in an aromatic cycle)
    The molecular plane is the one containing two vectors:
    i) From the centroid of the atoms to the first atom in the AtomGroup
    ii) from the centroid of the atoms to the third atom in the AtomGroup
    """
    center1, center2 = ag1.center_of_geometry(), ag2.center_of_geometry()
    v1a = ag1.atoms[0].position - center1
    v1b = ag1.atoms[2].position - center1
    norm1 = np.cross(v1a, v1b)
    link = center2 - center1
    angle = angle_between_vectors(norm1, link)
    angle = 180 - angle if angle > 90 else angle  # This angle is cyclic. 91 deg is the same as 89 deg
    return angle


def pistacking(U, props, sel):
    """
    Returns 3 arrays with the following information of all the pi stackings found:
    i) The inter-centroid distance of the stacked groups
    ii) Tilt angle, that is, the angle between the aromatic rings' planes (deg)
    iii) Phase (i.e. offset) angle between the aromatic rings. This is the angle between
    the plane of one ring and the vector connecting the centroids of the rings
    """
    g_ref = sel[props['ref']]
    g_target = sel[props['target']]
    g_ref_byres = list(g_ref.groupby('resids').values())
    g_target_byres = list(g_target.groupby('resids').values())
    DT = U.trajectory[0].dt
    n_read = int(props['stop_ps'] - props['start_ps']) / DT + 1
    all_dists, all_tilts, all_phases = [], [], []
    for ts in tqdm(U.trajectory, total=n_read):
        if ts.time > props['stop_ps']:
            break
        elif ts.time >= props['start_ps']:
            ref_cogs = g_ref.center_of_geometry(compound='residues')
            target_cogs = g_target.center_of_geometry(compound='residues')
            dists = cdist(ref_cogs, target_cogs)
            close_ix = np.where(dists < props['d_max'])
            close_dists = list(dists[close_ix])
            tilts = [angle_between_planes(g_ref_byres[ref_ix], g_target_byres[target_ix])
                     for ref_ix, target_ix in zip(*close_ix)]
            phases = [angle_between_plane_vector(
                g_ref_byres[ref_ix], g_target_byres[target_ix]) for ref_ix, target_ix in zip(*close_ix)]
            all_dists += close_dists
            all_tilts += tilts
            all_phases += phases
    all_dists, all_tilts, all_phases = np.array(all_dists), np.array(all_tilts), np.array(all_phases)
    """fig = plt.figure()
    ax = plt.axes()
    ax.scatter(all_tilts, all_phases, alpha=0.4, s=10)
    ax.set_xlabel('Tilts')
    ax.set_ylabel('Phases')
    plt.show()"""
    return all_dists / 10, all_tilts, all_phases


def pistacking_bound(U, props, sel):
    """
    Returns 3 arrays with the following information of all the pi stackings found:
    i) The inter-centroid distance of the stacked groups
    ii) Tilt angle, that is, the angle between the aromatic rings' planes (deg)
    iii) Phase (i.e. offset) angle between the aromatic rings. This is the angle between
    the plane of one ring and the vector connecting the centroids of the rings
    The data returns has the additional condition that there must be an H-H contact between
    the reference (e.g. monolayer) and target (e.g. analyte)
    """
    g_anchor = sel[props['anchor']]
    g_ref_pi = sel[props['ref_pi']]
    g_ref_pi_byres = list(g_ref_pi.groupby('resids').values())
    g_ref_H = sel[props['ref_H']]
    g_target_pi = sel[props['target_pi']]
    g_target_pi_byres = list(g_target_pi.groupby('resids').values())
    g_target_H = sel[props['target_H']]

    DT = U.trajectory[0].dt
    n_read = (props['stop_ps'] - props['start_ps']) // DT + 1
    all_dists, all_tilts, all_phases = [], [], []
    for ts in tqdm(U.trajectory, total=n_read):
        if ts.time >= props['stop_ps']:
            break
        elif ts.time >= props['start_ps']:
            ref_cogs = g_ref_pi.center_of_geometry(compound='residues')
            target_cogs = g_target_pi.center_of_geometry(compound='residues')
            for r, res in enumerate(g_target_H.residues):
                g_target_subH = res.atoms.intersection(g_target_H)
                dists = cdist(g_ref_H.positions, g_target_subH.positions)
                if np.any(dists <= props['d_max']):
                    dists_pi = cdist(ref_cogs, [target_cogs[r]])
                    close_ix = np.where(dists_pi < props['d_max_pi'])
                    close_dists = list(dists_pi[close_ix])
                    tilts = [angle_between_planes(g_ref_pi_byres[ref_ix], g_target_pi_byres[r])
                             for ref_ix, _ in zip(*close_ix)]
                    phases = [angle_between_plane_vector(
                        g_ref_pi_byres[ref_ix], g_target_pi_byres[r]) for ref_ix, _ in zip(*close_ix)]
                    all_dists += close_dists
                    all_tilts += tilts
                    all_phases += phases
    all_dists, all_tilts, all_phases = np.array(all_dists), np.array(all_tilts), np.array(all_phases)
    """fig = plt.figure()
    ax = plt.axes()
    ax.scatter(all_tilts, all_phases, alpha=0.4, s=10)
    ax.set_xlabel('Tilts')
    ax.set_ylabel('Phases')
    plt.show()"""
    return all_dists / 10, all_tilts, all_phases


def write_pistacking(dists, tilts, phases, props, name):
    """
    Writes a file describing all the pi stacking events
    The output includes the distance between centroids, the tilt angle, and phase (offset angle)
    """
    f = open(name + "_pistacking.sfu", 'w')
    f.write("#Distance between ring centroids, their tilting angle (i.e. angle between rings) \
    and their phase angle (i.e. displacemente between rings)\n")
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    f.write("#Distance between centroids (nm) \t Tilt angle (deg) \t Phase angle (deg)\n")
    for d, t, p in zip(dists, tilts, phases):
        f.write("{:<10.3f} {:>10.3f} {:>10.3f}\n".format(d, t, p))
    f.close()


def write_pistacking_bound(dists, tilts, phases, props, name):
    """
    Writes a file describing all the pi stacking events
    The output includes the distance between centroids, the tilt angle, and phase (offset angle)
    """
    f = open(name + "_pistackingbound.sfu", 'w')
    f.write("#Distance between ring centroids, their tilting angle (i.e. angle between rings) \
    and their phase angle (i.e. displacemente between rings). Data shown only for situation where \
    there is an H-H contact\n")
    for key, val in props.items():
        f.write("#{:<10}   {:<20}\n".format(key, str(val)))
    f.write("#Distance between centroids (nm) \t Tilt angle (deg) \t Phase angle (deg)\n")
    for d, t, p in zip(dists, tilts, phases):
        f.write("{:<10.3f} {:>10.3f} {:>10.3f}\n".format(d, t, p))
    f.close()


def pipeline_pistacking(U, props, sel, name):
    """
    Pipeline for calculating pi stacking events between two moieties
    """
    dists, tilts, phases = pistacking(U, props, sel)
    write_pistacking(dists, tilts, phases, props, name)


def pipeline_pistacking_bound(U, props, sel, name):
    """
    Pipeline for calculating pi stacking events between two moieties given that there is an H-H contact
    The H-H contact is not necessarily between the groups stacked
    """
    dists, tilts, phases = pistacking_bound(U, props, sel)
    write_pistacking_bound(dists, tilts, phases, props, name)
