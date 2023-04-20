"""Utility functions used here and there throughout the package."""
from typing import Any, Dict, Optional

import numpy as np
from MDAnalysis import AtomGroup, Universe
from MDAnalysis.core.groups import Atom
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
    ref_frame: int = 0,
    get_bulk_atoms: bool = False
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
        get_bulk_atoms (bool, optional):
            Whether to return the bulk atoms or the surface atoms.

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
    if get_bulk_atoms is False:
        return np_atom_group[n_neighbors == n_max_neighbors]
    return np_atom_group[n_neighbors < n_max_neighbors]


def get_n_neighbors(
    atom_group: AtomGroup,
    atom: Atom,
    distance_threshold: float
) -> int:
    """
    Determines the number of neighbors (i.e., atoms under a distance threshold) 
    of a group of atoms with respect to a reference atom.

    Args:
        atom_group (AtomGroup): 
            Atom group were to look for the neighbors. 
            The atom 'atom' can be included in the group.
        atom (Atom): 
            Atom for which to compute the number of neighbors.
        distance_threshold (float): 
            Maximum distance for a pair of atoms to be considered neighbors.

    Returns:
        int: 
            Number of neighbors that an atom has.
    """
    atom_group_without_atom = atom_group.subtract(atom)
    dists = cdist(atom_group_without_atom.positions,
                  atom.position[np.newaxis, ...])
    return np.sum(dists <= distance_threshold)


def get_atoms_with_n_neighbors(
    atom_group: AtomGroup,
    distance_threshold: float,
    n_neighbors: int
) -> AtomGroup:
    """
    Gets the subgroup of atoms with a certain number of neighbors.

    Args:
        atom_group (AtomGroup): 
            Atom group to subsample.
        distance_threshold (float): 
            Maximum distance for an atom-atom pair to be neighbors.
        n_neighbors (int): 
            Number of nighbors in each atom of the final atom group.

    Returns:
        AtomGroup: 
            Atom group with atoms that each hace n_neighbors neighbors.
    """
    atom_group_neighbors = np.array([get_n_neighbors(
        atom_group=atom_group, atom=atom, distance_threshold=distance_threshold)
        for atom in atom_group])
    return atom_group[atom_group_neighbors == n_neighbors]


def print_atom_group_as_vmd_prompt(atom_group: AtomGroup) -> None:
    """
    Prints a VMD selection with the index of all atoms in a group.

    Args:
        atom_group (AtomGroup): 
            Atom group to select in VMD.
    """
    prompt = "index "
    prompt += " ".join([str(atom.id) for atom in atom_group])
    print(prompt, end='\n\n')


def print_atom_group_as_gmx_group(atom_group: AtomGroup, group_name: Optional[str]) -> None:
    """
    Prints an MDAnalysis AtomGroup as a Gromacs group.

    Args:
        atom_group (AtomGroup): 
            MDAnalysis atom group.
        group_name (Optional[str]):
            Gromacs group name.
    """
    prompt = f"[ {group_name} ]\n"
    prompt += " ".join([str(atom.id + 1) for atom in atom_group])
    print(prompt, end="\n\n")
