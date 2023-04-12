"""Base class for managing input parameters of analysis metrics and their outputs"""
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from MDAnalysis import AtomGroup, Universe


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
            _ = [self.atom_groups[group_name] for group_name in group_names]
            return None
        except KeyError as key_error:
            print("Some atom groups could not be fetched."
                  "Are you sure you passed an atom_groups dictionary?")
            raise key_error

    def __str__(self) -> str:
        """
        Writes the class as a tabular string message.

        Returns:
            str: 
                Input parameters as a class.
        """
        written_input = ""
        for key, value in self.__dict__.items():
            if key == "atom_groups":
                continue
            if isinstance(value, AtomGroup):
                written_input += f"# {key:<25} <AtomGroup with {len(value)} atoms>\n"
                continue
            written_input += f"# {key:<25} {str(value):<100}\n"
        return written_input


class BaseOutput():
    """
    Base class with the outputs of an analysis function.
    """

    def __init__(self) -> None:
        """
        Initializer
        """
        return


class BaseWriter():
    """
    Base class for writing the inputs and outputs of an analysis run.
    """

    def __init__(
        self,
        analysis_input: BaseInput,
        analysis_output: BaseOutput,
    ) -> None:
        """
        Initializer.
        """
        self.analysis_input = analysis_input
        self.analysis_output = analysis_output

    def write(
        self,
        filename: Union[Path, str],
        universe: Universe
    ) -> None:
        """
        Writes the input and output data to a file.

        Args:
            filename (Union[Path, str]): 
                File where to save the data.
            universe (Universe):
                Universe object with the MD metadata.
        """
        with open(filename, "w", encoding="utf-8") as file:
            file.write(f"# TIME: {datetime.now()}\n")
            file.write(f"# TPR: {universe.filename}\n")
            file.write(f"# XTC: {universe.trajectory.filename}\n\n")
            file.write("# INPUT:\n")
            file.write(str(self.analysis_input))
            file.write("\n# OUTPUT:\n")
            file.write(str(self.analysis_output))
