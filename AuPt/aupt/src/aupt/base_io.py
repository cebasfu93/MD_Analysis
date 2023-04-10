"""Base class for managing input parameters of analysis metrics and their outputs"""
from typing import Dict, List, Optional

from MDAnalysis import AtomGroup


class BaseInput:
    """
    Base class with input parameters shared by most (if not all) analysis.
    """

    def __init__(
        self,
        atom_groups: Optional[Dict[str, AtomGroup]] = None,
        start_time: float = 0,
        stop_time: Optional[float] = None
    ) -> None:
        """
        Initializer

        Args:
            atom_groups (Dict[str, AtomGroup]): 
                Dictionary with the reference/target groups label and the corresponding atom group.
            start_ps (float): 
                Time in ps where to start analyzing the trajectory.
            stop_ps (Optional[float]): _description_
                Time in ps where to end analyzing the trajectory.
                If None, the analyzes goes until the end of the trajectory, 
                but this is implemented elsewhere.
        """
        self.atom_groups: Dict[str, AtomGroup] = atom_groups or {}
        self.start_time: float = start_time
        self.stop_time: Optional[float] = stop_time

    def validate_atom_groups(self, group_names: List[str]) -> None:
        """
        Tries to fetch all keys from a dictionary of atom groups.

        Args:
            atom_groups (Dict[str, AtomGroup]):
                Dictionary with a label as key and an AtomGroup as value.
            group_names (List[str]):
                List of group names/labels to validate.
        Raises:
            KeyError:
                If some atom group is not present in the atom groups dictionary.
        """
        try:
            print(group_names)
            print(self.atom_groups)
            _ = [self.atom_groups[group_name] for group_name in group_names]
            return None
        except KeyError as key_error:
            print("Some atom groups could not be fetched."
                  "Are you sure you passed an atom_groups dictionary?")
            raise key_error


class BaseOutput():
    """
    Base class with the outputs of an analysis function.
    """

    def __init__(self) -> None:
        """
        Initializer
        """
        return
