"""Module with classes that handle arguments of analysis metrics that are averaged over time."""
from typing import List, Tuple

from MDAnalysis import AtomGroup

from aupt.base_io import BaseInput


class RadialDistributionFunctionsInput(BaseInput):
    """
    Class with inputs to compute radial distributions functions.
    """

    def __init__(
        self,
        r_range: Tuple[float, float],
        nbins: int,
        ref_group_name: str,
        target_groups_name: List[str],
        **kwargs
    ) -> None:
        """
        Initializer.

        Args:
            r_range (Tuple[float, float]): 
                Range of distances (from the reference group COM in A) for which to compute RDFs.
            nbins (int): 
                Number of bins on which to split r_range. 
                The number of points in the result is nbins - 1
            ref_group_name (str): 
                Label of the atom group to use as reference.
            target_groups_name (List[str]): 
                Labels of the atom groups to use as target.
        """
        super().__init__(**kwargs)

        group_names = target_groups_name + [ref_group_name]
        self.validate_atom_groups(group_names=group_names)

        self.r_range: Tuple[float, float] = r_range
        self.nbins: int = nbins
        self.ref_group_name: str = ref_group_name
        self.target_groups_name: List[str] = target_groups_name

        self.ref_group: AtomGroup = self.atom_groups[ref_group_name]
        self.target_groups: List[AtomGroup] = [
            self.atom_groups[name] for name in self.target_groups_name]


class BindingTimeInput(BaseInput):
    """
    Class with inputs to compute radial distributions functions.
    """

    def __init__(
            self,
            group1_name: str,
            group1_com: bool,
            group2_name: str,
            group2_com: bool,
            distance_threshold: float,
            **kwargs) -> None:

        super().__init__(**kwargs)

        group_names = [group1_name, group2_name]
        self.validate_atom_groups(group_names=group_names)

        self.group1_name: str = group1_name
        self.group1_com: bool = group1_com
        self.group2_name: str = group2_name
        self.group2_com: bool = group2_com
        self.distance_threshold: float = distance_threshold
