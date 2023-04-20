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
        ref_group_surf: bool,
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
            ref_group_surf (bool):
                Whether to compute the RDF with respect to the closest atom to the reference group.
            target_groups_name (List[str]): 
                Labels of the atom groups to use as target.
        """
        super().__init__(**kwargs)

        group_names = target_groups_name + [ref_group_name]
        self.validate_atom_groups(group_names=group_names)

        self.r_range: Tuple[float, float] = r_range
        self.nbins: int = nbins
        self.ref_group_name: str = ref_group_name
        self.ref_group_surf: bool = ref_group_surf
        self.target_groups_name: List[str] = target_groups_name

        self.ref_group: AtomGroup = self.atom_groups[ref_group_name]
        self.target_groups: List[AtomGroup] = [
            self.atom_groups[name] for name in self.target_groups_name]


class BindingTimesInput(BaseInput):
    """
    Class with inputs to compute radial distributions functions.
    """

    def __init__(
            self,
            ref_group_name: str,
            ref_group_com: bool,
            target_group_name: str,
            target_group_rescom: bool,
            distance_threshold: float,
            **kwargs) -> None:
        """
        Initializer.

        Args:
            ref_group_name (str):
                Label of the reference group. 
            ref_group_com (bool): 
                Whether a contact is defined by 
                the distance to the COM of the reference group (True) 
                or any atom in the reference group (False).
            target_group_name (str): 
                Label of the target group.
            target_group_rescom (bool): 
                Whether a contact is defined by 
                the distance to the COM of a residue in the reference group (True) 
                or any atom in a given residue of the target group (False).
            distance_threshold (float): 
                Maximum distance (in A) at which two atoms (or COMs) 
                are considered to be in contact.
        """
        super().__init__(**kwargs)

        group_names = [ref_group_name, target_group_name]
        self.validate_atom_groups(group_names=group_names)

        self.ref_group_name: str = ref_group_name
        self.ref_group_com: bool = ref_group_com
        self.ref_group: AtomGroup = self.atom_groups[self.ref_group_name]
        self.target_group_name: str = target_group_name
        self.target_group_rescom: bool = target_group_rescom
        self.target_group: AtomGroup = self.atom_groups[self.target_group_name]
        self.distance_threshold: float = distance_threshold
