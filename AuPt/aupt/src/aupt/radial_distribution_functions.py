from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from MDAnalysis import AtomGroup, Universe
from scipy.integrate import cumtrapz
from tqdm import tqdm

FloatArray = npt.NDArray[np.float32]


def rdf(
        universe: Universe,
        start_time: float,
        stop_time: float,
        r_range: Tuple[float, float],
        nbins: int,
        atom_groups: Dict[str, AtomGroup],
        ref_group_name: str,
        target_groups_name: List[str]) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """
    Calculates radial distribution function with respect to the center of mass of a reference group.
    The RDF of the solvent does not converge to 1 if the reference group (e.g., a nanoparticle)
    has a non-negligible volume compared to the simulation box.

    Args:
        universe (Universe): 
            Universe object with the data from the MD simulation.
        start_time (float): 
            Start time (in ps) to analyze.
        stop_time (float): 
            Stop time (in ps) to analyze.
        r_range (Tuple[float, float]): 
            Minimum and maximum distances (from the reference group's COM) to look at.
        nbins (int): 
            Number of bins in which to split the data. The result will have nbins - 1 data points.
        atom_groups (Dict[str, AtomGroup]): 
            Dictionary where the values are atom groups from the MD simulation.
        ref_group_name (str): 
            Name of atom group to use as reference.
        target_groups_name (List[str]): 
            Name of the atom groups for which to compute the RDF.

    Returns:
        Tuple[FloatArray, FloatArray, FloatArray]: 
            Middle of the binds, RDF of the target groups, and cumulative RDF of the target groups.
    """
    delta_t = universe.trajectory[0].dt
    n_read = int(stop_time - start_time) / delta_t + 1
    rdfs = {}
    rdfs_cum = {}
    r_space = np.linspace(r_range[0], r_range[1], nbins)
    r_center = 0.5 * (r_space[1:] + r_space[:-1])
    g_ref = atom_groups[ref_group_name]
    print(f"Reference COM: {ref_group_name}")
    for target_name in target_groups_name:
        n_frames = 0
        print(f"Current target: {target_name}")
        counts = np.zeros(len(r_space) - 1)
        g_target = atom_groups[target_name]
        for ts in tqdm(universe.trajectory, total=n_read):
            if ts.time > stop_time:
                break
            elif ts.time >= start_time:
                n_frames += 1
                x_ref = g_ref.center_of_mass()
                x_target = g_target.positions
                dx = np.subtract(x_target, x_ref)
                dists = np.linalg.norm(dx, axis=1)
                dists = dists[dists <= r_range[1]]
                counts += np.histogram(dists, bins=r_space)[0]
        # Average particles counted in target group
        n_target = np.sum(counts) / n_frames
        v_shells = 4 * np.pi * (r_space[1:] ** 3 - r_space[:-1] ** 3) / 3
        homo_dens = 3 * n_target / (4 * np.pi * r_space[-1] ** 3)
        counts /= n_frames  # Averages number of target particles at each bin over time
        # Divide on the number of reference particles (1 center of mass)
        counts /= 1
        counts /= v_shells  # Divide on shell volume
        counts /= homo_dens  # Divide on homogeneous density
        rdfs[target_name] = counts
        integrand = 4 * np.pi * np.multiply(r_center**2, counts)
        # returns a shape with one element less
        cumulative = homo_dens * cumtrapz(y=integrand, x=r_center, initial=0.0)
        rdfs_cum[target_name] = cumulative
    return r_center, rdfs, rdfs_cum
