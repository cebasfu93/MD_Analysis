from MDAnalysis import AtomGroup
from aupt.base_io import BaseInput


def test_empty_init():
    x = BaseInput()
    assert x.atom_groups == dict()
    assert x.start_time == 0
    assert x.stop_time is None


def test_init(universe):
    mock_group_name = "my_group"
    x = BaseInput(
        atom_groups={mock_group_name: AtomGroup([], universe)},
        start_time=10,
        stop_time=20)
    assert mock_group_name in x.atom_groups
    assert x.atom_groups[mock_group_name].n_atoms == 0
    assert x.start_time == 10
    assert x.stop_time == 20
