"""Module to compute number of contacts over time."""
from typing import List

import numpy as np
from MDAnalysis import AtomGroup, Universe
from scipy.spatial.distance import cdist
from tqdm import tqdm

from aupt.time_series.inputs import ContactsNumberInput
from aupt.time_series.outputs import ContactsNumberOutput
from aupt.utils import get_number_of_frames_to_read


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
            Object with the input parameters needed to compute binding times.

    Returns:
        ContactsNumberOutput: 
            Analyzed time points, 
            number of atom-atom contacts,
            and number of residues with an atom-atom contact.
    """

    delta_t = universe.trajectory.dt
    n_read = get_number_of_frames_to_read(
        start_time=input_control.start_time,
        stop_time=input_control.stop_time,
        delta_t=delta_t
    )

    n_target_groups = len(input_control.target_groups_name)
    time_points = np.zeros(n_read)
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

        for j, frame in tqdm(enumerate(universe.trajectory), total=n_read):
            if frame.time > input_control.stop_time:
                break
            if frame.time >= input_control.start_time:
                time_points[i] = frame.time
                x_ref = input_control.ref_group.positions
                x_targets = [r.positions for r in target_group_residues]
                dists = [cdist(x_ref, x_target) for x_target in x_targets]
                contacts_masks = [
                    dist_matrix <= input_control.distance_threshold for dist_matrix in dists]

                n_contacts_now = sum(np.sum(mask) for mask in contacts_masks)
                n_contacts[i, j] = n_contacts_now

                n_residue_contacts_now = sum(np.any(mask)
                                             for mask in contacts_masks)
                n_residue_contacts[i, j] = n_residue_contacts_now

    n_contacts_wrapped = dict(
        zip(input_control.target_groups_name, n_contacts)
    )
    n_residue_contacts_wrapped = dict(
        zip(input_control.target_groups_name, n_residue_contacts)
    )

    return ContactsNumberOutput(time_points=time_points,
                                number_of_contacts=n_contacts_wrapped,
                                number_of_contact_residues=n_residue_contacts_wrapped)
