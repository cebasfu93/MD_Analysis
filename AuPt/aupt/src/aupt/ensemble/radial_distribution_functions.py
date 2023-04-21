"""Module to compute radial distribution functions."""
import numpy as np
from MDAnalysis import Universe
from scipy.integrate import cumtrapz
from scipy.spatial.distance import cdist
from tqdm import tqdm

from aupt.ensemble.inputs import RadialDistributionFunctionsInput
from aupt.ensemble.outputs import RadialDistributionFunctionsOutput
from aupt.utils import get_number_of_frames_to_read


def rdf(
    universe: Universe,
    input_control: RadialDistributionFunctionsInput,
) -> RadialDistributionFunctionsOutput:
    """
    Calculates radial distribution function with respect to the center of mass of a reference group.
    The RDF of the solvent does not converge to 1 if the reference group (e.g., a nanoparticle)
    has a non-negligible volume compared to the simulation box.

    Args:
        universe (Universe): 
            Universe object with the data from the MD simulation.
        input_control (RadialDistributionFunctions):
            Object with the input parameters needed to compute 
            radial distribution functions.

    Returns:
        RadialDistributionFunctionsOutput: 
            Middle of the binds, RDF of the target groups, and cumulative RDF of the target groups.
    """
    # TODO: Use the COM of each molecule in each target, the COM of each target groups, or the atoms
    # Right now it supports only the latter
    delta_t = universe.trajectory.dt
    n_read = get_number_of_frames_to_read(
        start_time=input_control.start_time,
        stop_time=input_control.stop_time,
        delta_t=delta_t)
    rdfs = {}
    rdfs_cum = {}
    r_space = np.linspace(
        input_control.r_range[0],
        input_control.r_range[1],
        input_control.nbins)
    r_center = 0.5 * (r_space[1:] + r_space[:-1])
    print(f"Reference COM: {input_control.ref_group_name}")
    for target_group_name, target_group in \
            zip(input_control.target_groups_name, input_control.target_groups):
        n_frames = 0
        print(f"Current target: {target_group_name}")
        counts = np.zeros(len(r_space) - 1, dtype="float")
        for frame in tqdm(universe.trajectory, total=n_read):
            if frame.time > input_control.stop_time:
                break
            if frame.time >= input_control.start_time:
                n_frames += 1
                x_target = target_group.positions
                if input_control.ref_group_surf is False:
                    x_ref = input_control.ref_group.center_of_mass()
                    delta_x = np.subtract(x_target, x_ref)
                    dists = np.linalg.norm(delta_x, axis=1)
                    dists = dists[dists <= input_control.r_range[1]]
                else:
                    x_ref = input_control.ref_group.positions
                    dists = cdist(x_ref, x_target)
                    dists = np.min(dists, axis=0)
                counts += np.histogram(dists, bins=r_space)[0]
        # Average particles counted in target group
        n_target = np.sum(counts) / n_frames
        v_shells = 4 * np.pi * (r_space[1:] ** 3 - r_space[:-1] ** 3) / 3
        homo_dens = 3 * n_target / (4 * np.pi * r_space[-1] ** 3)
        counts /= n_frames  # Averages number of target particles at each bin over time
        # Divide on the number of reference particles (1 center of mass)
        counts /= 1 if input_control.ref_group_surf is False \
            else len(input_control.ref_group)
        counts /= v_shells  # Divide on shell volume
        counts /= homo_dens  # Divide on homogeneous density
        rdfs[target_group_name] = counts
        integrand = 4 * np.pi * np.multiply(r_center**2, counts)
        # returns a shape with one element less
        cumulative = homo_dens * cumtrapz(y=integrand, x=r_center, initial=0.0)
        rdfs_cum[target_group_name] = cumulative
    return RadialDistributionFunctionsOutput(space=r_space,
                                             rdfs=rdfs,
                                             cumulative_rdfs=rdfs_cum)
