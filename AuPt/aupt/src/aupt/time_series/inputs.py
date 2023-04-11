"""Module with classes that handle arguments of time series analysis metrics."""
from MDAnalysis import AtomGroup
from aupt.base_io import BaseInput


class ContactsNumberInput(BaseInput):
    """
    Class with inputs to compute number of contacts over time.
    """

    def __init__(
        self,
        ref_group_name: str,
        target_group_name: str,
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
            target_group_name (str): 
                Name of the target group. 
                All unique residues in this group are extracted 
                and then the atoms of each residue are processed. 
                No COM is extracted.
            distance_threshold (float): 
                Maximum distance (in A) between 2 atoms considered as a contact.
        """

        super().__init__(**kwargs)

        group_names = [ref_group_name, target_group_name]
        self.validate_atom_groups(group_names=group_names)

        self.distance_threshold: float = distance_threshold
        self.ref_group_name: str = ref_group_name
        self.target_group_name: str = target_group_name
        self.ref_group: AtomGroup = self.atom_groups[self.ref_group_name]
        self.target_group: AtomGroup = self.atom_groups[self.target_group_name]
