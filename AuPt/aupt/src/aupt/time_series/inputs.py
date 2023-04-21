"""Module with classes that handle arguments of time series analysis metrics."""
from typing import List
from MDAnalysis import AtomGroup
from aupt.base_io import BaseInput


class ContactsNumberInput(BaseInput):
    """
    Class with inputs to compute number of contacts over time.
    """

    def __init__(
        self,
        ref_group_name: str,
        target_groups_name: List[str],
        distance_threshold: float,
        **kwargs
    ) -> None:
        """
        Initializer.

        Args:
            ref_group_name (str): 
                Name of the reference group. 
                All atoms in the reference group are considered individually, 
                i.e., no COM is extracted.
            target_groups_name (List[str]): 
                Name of the target groups. 
                For each group, all unique residues are extracted 
                and then the atoms of each residue are processed. 
                No COM is extracted.
            distance_threshold (float): 
                Maximum distance (in A) between 2 atoms considered as a contact.
        """

        super().__init__(**kwargs)

        group_names = target_groups_name + [ref_group_name]
        self.validate_atom_groups(group_names=group_names)

        self.distance_threshold: float = distance_threshold
        self.ref_group_name: str = ref_group_name
        self.target_groups_name: List[str] = target_groups_name
        self.ref_group: AtomGroup = self.atom_groups[self.ref_group_name]
        self.target_groups: List[AtomGroup] = [
            self.atom_groups[name] for name in self.target_groups_name
        ]


class SaltBridgesNumberInput(BaseInput):
    """
    Class with inputs to compute number of salt bridges over time.
    """

    def __init__(
        self,
        cations_group_name: str,
        anions_group_name: str,
        distance_threshold: float,
        **kwargs
    ) -> None:
        """
        Initializer.

        Args:
            cations_group_name (str): 
                Name of the group with the cations (only). 
            anions_group_name (str): 
                Name of the group with the anions (only). 
            distance_threshold (float): 
                Maximum distance (in A) between 2 atoms considered as a salt bridge.
        """

        super().__init__(**kwargs)

        group_names = [cations_group_name, anions_group_name]
        self.validate_atom_groups(group_names=group_names)

        self.distance_threshold: float = distance_threshold
        self.anions_group_name: str = anions_group_name
        self.cations_group_name: str = cations_group_name
        self.anions_group: AtomGroup = self.atom_groups[self.anions_group_name]
        self.cations_group: AtomGroup = self.atom_groups[self.cations_group_name]


class HydrogenBondsInput(BaseInput):
    """
    Class with inputs to compute number of hydrogen bonds over time.
    """

    def __init__(
        self,
        donors_sel: str,
        hydrogens_sel: str,
        acceptors_sel: str,
        d_a_cutoff: float,
        d_h_cutoff: float,
        d_h_a_angle_cutoff: float,
        update_selections: bool,
        **kwargs
    ) -> None:
        """
        Initializer.

        Args:
            donors_sel (str): 
                Selection text for the H-bond donors.
            hydrogens_sel (str):
                Selection text for the hydrogen atoms involved in H-bonds.
            acceptors_sel (str): 
                Selection text for the H-bond acceptors.
            d_a_cutoff (float): 
                Donor-acceptor maximum distance (A).
            d_h_cutoff (float): 
                Donor-hydrogen maximum distance (A).
            d_h_a_angle_cutoff (float): 
                Donor-hydrogen-acceptor minimum angle (degrees).
            update_selections (bool): 
                Whether the selections should be updated every frame or not.
        """
        super().__init__(**kwargs)

        self.donors_sel: str = donors_sel
        self.hydrogens_sel: str = hydrogens_sel
        self.acceptors_sel: str = acceptors_sel
        self.d_a_cutoff: float = d_a_cutoff
        self.d_h_cutoff: float = d_h_cutoff
        self.d_h_a_angle_cutoff: float = d_h_a_angle_cutoff
        self.update_selections: bool = update_selections
