"""Module to compute number of contacts over time."""
from typing import List

import numpy as np
from MDAnalysis import AtomGroup, Universe
from scipy.spatial.distance import cdist
from tqdm import tqdm

from aupt.time_series.inputs import ContactsNumberInput, SaltBridgesNumberInput
from aupt.time_series.outputs import ContactsNumberOutput, SaltBridgesOutput
from aupt.utils import get_time_points_in_universe


def number_of_contacts(
    universe: Universe,
    input_control: ContactsNumberInput
) -> ContactsNumberOutput:
    """
    Calculates the number of atom-atom contacts
    and the number of (target) residues making an atom-atom contact
    with a reference group.

    Args:
        universe (Universe): 
            Universe object with the data from the MD simulation.
        input_control (ContactsNumberInput): 
            Object with the input parameters needed to compute the number of contacts.

    Returns:
        ContactsNumberOutput: 
            Analyzed time points, 
            number of atom-atom contacts,
            and number of residues with an atom-atom contact.
    """
    time_points = get_time_points_in_universe(
        start_time=input_control.start_time,
        stop_time=input_control.stop_time,
        universe=universe
    )
    n_read = len(time_points)
    n_tqdm = int((min(universe.trajectory[-1].time, input_control.stop_time) -
                  max(0, universe.trajectory[0].time)) / universe.trajectory.dt) + 1

    n_target_groups = len(input_control.target_groups_name)
    n_contacts = np.zeros((n_target_groups, n_read), dtype='int')
    n_residue_contacts = np.zeros(
        (n_target_groups, n_read),
        dtype='int'
    )

    print(f"Reference group: {input_control.ref_group_name}")
    for i, (target_group, target_group_name) in \
            enumerate(zip(input_control.target_groups, input_control.target_groups_name)):
        print(f"Current target: {target_group_name}")
        target_group_residues: List[AtomGroup] = target_group.split('residue')
        read_frame_ndx = 0
        for frame in tqdm(universe.trajectory, total=n_tqdm):
            if frame.time > input_control.stop_time:
                break
            if frame.time >= input_control.start_time:
                x_ref = input_control.ref_group.positions
                x_targets = [r.positions for r in target_group_residues]
                dists = [cdist(x_ref, x_target) for x_target in x_targets]
                contacts_masks = [
                    dist_matrix <= input_control.distance_threshold for dist_matrix in dists]

                n_contacts_now = sum(np.sum(mask) for mask in contacts_masks)
                n_contacts[i, read_frame_ndx] = n_contacts_now

                n_residue_contacts_now = sum(np.any(mask)
                                             for mask in contacts_masks)
                n_residue_contacts[i, read_frame_ndx] = n_residue_contacts_now
                read_frame_ndx += 1

    n_contacts_wrapped = dict(
        zip(input_control.target_groups_name, n_contacts)
    )
    n_residue_contacts_wrapped = dict(
        zip(input_control.target_groups_name, n_residue_contacts)
    )

    return ContactsNumberOutput(time_points=time_points,
                                number_of_contacts=n_contacts_wrapped,
                                number_of_contact_residues=n_residue_contacts_wrapped)


def number_of_salt_bridges(
    universe: Universe,
    input_control: SaltBridgesNumberInput
) -> SaltBridgesOutput:
    """
    Calculates the number of salt bridges, i.e., contacts between anions and cations.

    Args:
        universe (Universe): 
            Universe object with the data from the MD simulation.
        input_control (ContactsNumberInput): 
            Object with the input parameters needed to compute number of salt bridges.

    Returns:
        SaltBridgesOutput: 
            Analyzed time points, 
            number of salt bridges
    """
    time_points = get_time_points_in_universe(
        start_time=input_control.start_time,
        stop_time=input_control.stop_time,
        universe=universe
    )
    n_read = len(time_points)
    n_tqdm = int((min(universe.trajectory[-1].time, input_control.stop_time) -
                  max(0, universe.trajectory[0].time)) / universe.trajectory.dt) + 1
    n_bridges = np.zeros(n_read, dtype='int')

    print(f"Anions group: {input_control.anions_group_name}")
    print(f"Cations group: {input_control.cations_group_name}")
    read_frame_ndx = 0
    for frame in tqdm(universe.trajectory, total=n_tqdm):
        if frame.time > input_control.stop_time:
            break
        if frame.time >= input_control.start_time:
            x_anion = input_control.anions_group.positions
            x_cation = input_control.cations_group.positions
            dists = cdist(x_anion, x_cation)
            n_bridges[read_frame_ndx] = np.sum(dists <= input_control.distance_threshold)
            read_frame_ndx += 1

    return SaltBridgesOutput(time_points=time_points,
                            number_salt_bridges=n_bridges)
