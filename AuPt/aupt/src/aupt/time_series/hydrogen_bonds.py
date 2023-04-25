"""Module to compute number of hydrogen bonds over time."""
from MDAnalysis import Universe
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import \
    HydrogenBondAnalysis as HBA

from aupt.time_series.inputs import HydrogenBondsInput
from aupt.time_series.outputs import HydrogenBondsOutput
from aupt.utils import get_time_points_in_universe


def hydrogen_bonds(
    universe: Universe,
    input_control: HydrogenBondsInput
) -> HydrogenBondsOutput:
    """
    Calculates the number of hydrogen bonds over time.

    Args:
        universe (Universe): 
            Universe object with the data from the MD simulation.
        input_control (ContacHydrogenBondsInputtsNumberInput): 
            Object with the input parameters needed to compute hydrogen bonds.

    Returns:
        HydrogenBondsOutput: 
            Analyzed time points, 
            number of hydrogen bonds.
    """
    delta_t = universe.trajectory.dt

    hbonds = HBA(
        universe=universe,
        donors_sel=input_control.donors_sel,
        hydrogens_sel=input_control.hydrogens_sel,
        acceptors_sel=input_control.acceptors_sel,
        d_a_cutoff=input_control.d_a_cutoff,
        d_h_cutoff=input_control.d_h_cutoff,
        d_h_a_angle_cutoff=input_control.d_h_a_angle_cutoff,
        update_selections=input_control.update_selections
    )
    time_points = get_time_points_in_universe(
        start_time=input_control.start_time,
        stop_time=input_control.stop_time,
        universe=universe
    )
    start_frame = int(time_points[0] // delta_t)
    stop_frame = int(time_points[-1] // delta_t + 1)
    hbonds_output = hbonds.run(
        start=start_frame,
        stop=stop_frame,
        verbose=True
    )
    number_hbonds = hbonds_output.count_by_time().astype('int')
    return HydrogenBondsOutput(time_points=time_points,
                               number_hbonds=number_hbonds)
