"""Module to compute binding times between molecular entities."""
from typing import List

import numpy as np
from MDAnalysis import AtomGroup, Universe
from scipy.spatial.distance import cdist
from tqdm import tqdm

from aupt.ensemble.inputs import BindingTimesInput
from aupt.ensemble.outputs import BindingTimesOutput

from aupt.utils import get_number_of_frames_to_read


def binding_time(
    universe: Universe,
    input_control: BindingTimesInput
) -> BindingTimesOutput:
    """
    Calculates the time that two molecular entities spend close to each other.

    Args:
        universe (Universe): 
            Universe object with the data from the MD simulation.
        input_control (RadialDistributionFunctions):
            Object with the input parameters needed to compute 
            binding times.

    Returns:
        BindingTimesOutput: 
            Start and stop time (in ps) of each binding event, duration of each binding event, 
            and residue number of the target residue that is bound to the reference group.

    Raises:
            RuntimeError:
                If target_group_rescom is True
    """
    delta_t = universe.trajectory.dt
    n_read = get_number_of_frames_to_read(
        start_time=input_control.start_time,
        stop_time=input_control.stop_time,
        delta_t=delta_t)

    target_group_split: List[AtomGroup] = input_control.target_group.split(
        'residue')
    target_residue_numbers_split: List[int] = [
        g[0].resnum for g in target_group_split]

    n_targets = len(target_group_split)
    time_points = np.zeros(n_read)
    frames_array = np.zeros(n_read, dtype='int')
    # start and end with all contacts off
    bound_grid = np.zeros((n_targets, n_read + 2), dtype='int')
    for j, frame in tqdm(enumerate(universe.trajectory, 1), total=n_read):
        if frame.time > input_control.stop_time:
            break
        if frame.time >= input_control.start_time:
            time_points[j-1] = frame.time
            frames_array[j-1] = frame.frame
            x_ref = input_control.ref_group.positions\
                if input_control.ref_group_com is False\
                else input_control.ref_group.center_of_mass()[..., np.newaxis]
            x_targets = [res.positions for res in target_group_split]\
                if input_control.target_group_rescom is False\
                else [res.center_of_mass() for res in target_group_split]
            dists = [cdist(x_ref, x_target) for x_target in x_targets]
            target_in_contact = [
                np.any(dist < input_control.distance_threshold) for dist in dists
            ]
            bound_grid[target_in_contact, j] = 1  # there is contact
    transitions = bound_grid[:, 1:] - bound_grid[:, :-1]
    on_transitions = np.where(transitions == 1)
    off_transitions = np.where(transitions == -1)

    binding_times = np.zeros(len(on_transitions[0]))
    target_residue_numbers = np.zeros_like(binding_times)

    start_times = time_points[[on_ndx for on_ndx in on_transitions[1]]]
    stop_times = time_points[[off_ndx - 1 for off_ndx in off_transitions[1]]]
    start_frames = frames_array[[on_ndx for on_ndx in on_transitions[1]]]
    stop_frames = frames_array[[off_ndx - 1 for off_ndx in off_transitions[1]]]
    binding_times = stop_times - start_times
    target_residue_numbers = np.array([target_residue_numbers_split[r]
                                       for r in on_transitions[0]]).astype('int')

    return BindingTimesOutput(binding_times=binding_times,
                              start_times=start_times,
                              stop_times=stop_times,
                              start_frames=start_frames,
                              stop_frames=stop_frames,
                              target_residue_numbers=target_residue_numbers)
