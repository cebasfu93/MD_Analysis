from MDAnalysis import AtomGroup
from aupt.ensemble.inputs import RadialDistributionFunctionsInput


def test_radial_distribution_functions(universe):
    input_control = RadialDistributionFunctionsInput(
        atom_groups={"my_ref": AtomGroup([], universe)},
        start_time=0,
        r_range=(0, 10),
        nbins=50,
        ref_group_name="my_ref",
        target_groups_name=[]
    )
    assert input_control.start_time == 0
