"""Utility functions used here and there throughout the package."""
from typing import Any, Dict

import numpy as np
from MDAnalysis import AtomGroup, Universe
from scipy.spatial.distance import cdist


def update_dictionary_with_key_control(
    dict1: Dict[Any, Any],
    dict2: Dict[Any, Any],
    override: bool = False
) -> Dict[Any, Any]:
    """
    Updates a dictonary with another dictionary if there are non overlapping keys 
    or if override is True.

    Args:
        dict1 (Dict[Any, Any]): 
            Base dictionary to override.
        dict2 (Dict[Any, Any]): 
            Dictionary to override with.
        override (bool, optional): 
            Whether to override the overlapping keys or not.

    Returns:
        Dict[Any, Any]: _description_
    """
    overlapping_keys = dict1.keys() & dict2.keys()
    if override is True or len(overlapping_keys) == 0:
        return dict1.update(dict2)
    raise ValueError(
        "Could not update the dictionary "
        f"because there are overlapping keys: {overlapping_keys}")


def get_number_of_frames_to_read(
    start_time: float,
    stop_time: float,
    delta_t: float
) -> int:
    """
    Determines the number of frames to iterate over given 
    the start and end time of the simulation and the timestep.

    Args:
        start_time (float): 
            Start time to analyze (in ps).
        stop_time (float): 
            Stop time to analyze (in ps).
        delta_t (float): 
            Timestep of the simulation (in ps).

    Returns:
        int: 
            Number of frames to read.
    """
    return int(stop_time // delta_t - start_time // delta_t) + 1


def get_surface_atoms(
    universe: Universe,
    np_atom_group: AtomGroup,
    bond_distance: float,
    ref_frame: int = 0
) -> AtomGroup:
    """
    Extracts the atoms from a nanoparticle that don't have the maximal number of neighbors.

    Args:
        universe (Universe): 
            Universe of the MD simulation
        np_atom_group (AtomGroup): 
            Atom group with the nanoparticle (or crystal lattice in general).
        bond_distance (float): 
            Threshold distance (in A) under which an atom is considered a neighbor.
        ref_frame (int, optional): 
            Frame at which to check for neighbors.

    Returns:
        AtomGroup: 
            Atoms in a crystal lattice with a non-maximal number of neighbors.
    """
    universe.trajectory[ref_frame]
    dists = cdist(np_atom_group.positions, np_atom_group.positions)
    # remove the same atom (dist = 0)
    n_neighbors = np.sum(dists < bond_distance, axis=0) - 1
    n_max_neighbors = np.max(n_neighbors)
    print(f"Bulk atoms have {n_max_neighbors} neighbors")
    return np_atom_group[n_neighbors < n_max_neighbors]
