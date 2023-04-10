import os

import pytest
from MDAnalysis import Universe


@pytest.fixture(scope="session")
def universe():
    current_rel_path = os.path.dirname(os.path.realpath(__file__))
    xtc_path = os.path.join(
        current_rel_path,
        'test_md/traj.xtc',
    )
    tpr_path = os.path.join(
        current_rel_path,
        'test_md/traj.tpr',
    )
    return Universe(tpr_path, xtc_path)
