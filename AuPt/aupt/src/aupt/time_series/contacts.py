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

    delta_t = universe.trajectory.dt
    n_read = get_number_of_frames_to_read(
        start_time=input_control.start_time,
        stop_time=input_control.stop_time,
        delta_t=delta_t
    )

    time_points = np.zeros(n_read)
    n_contacts = np.zeros_like(time_points)

    target_group_residues: List[AtomGroup] = input_control.target_group.split(
        'residue')

    for i, frame in tqdm(enumerate(universe.trajectory), total=n_read):
        if frame.time > input_control.stop_time:
            break
        if frame.time > - input_control.start_time:
            time_points[i] = frame.time
            x_ref = input_control.ref_group.positions
            x_targets = [r.positions for r in target_group_residues]
            dists = [cdist(x_ref, x_target) for x_target in x_targets]
            n_contacts_now = sum(np.any(dist_matrix <= input_control.distance_threshold)
                                 for dist_matrix in dists)
            n_contacts[i] = n_contacts_now

    return ContactsNumberOutput(time_points=time_points,
                                number_of_contacts=n_contacts)
