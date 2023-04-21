"""Module with classes that handle the outputs of the temporal analysis functions."""
from dataclasses import dataclass
from typing import Dict
import numpy as np

from aupt.base_io import BaseOutput
from aupt.types import FloatArray, IntArray


@dataclass
class ContactsNumberOutput(BaseOutput):
    """
    Class with the outputs of a Contacts Number calculation.
    """
    time_points: FloatArray
    number_of_contacts: Dict[str, IntArray]
    number_of_contact_residues: Dict[str, IntArray]

    def __str__(self) -> str:
        """
        Writes the dataclass into a tabular string.

        Returns:
            str: 
                Output of the analysis as a tabular string.
        """
        labels = self.number_of_contacts.keys()
        assert labels == self.number_of_contact_residues.keys()  # security check

        unzipped_data = np.hstack([np.vstack(
            (self.number_of_contacts[label], self.number_of_contact_residues[label])).T
            for label in labels]).astype('int')
        n_data_cols = 2 * len(labels)
        written_output = f"{'time (ps)':<10} "
        for label in labels:
            written_output += f"{label + ' (AA)':<15} {label + ' (res)':<15} "
        written_output += "\n"
        for time_point, data in zip(self.time_points, unzipped_data):
            written_output += f"{time_point:<10.2f} " + \
                ("{:<15} " * n_data_cols).format(*data) + "\n"
        return written_output


@dataclass
class SaltBridgesOutput(BaseOutput):
    """
    Class with the outputs of a Salt Bridges calculation.
    """
    time_points: FloatArray
    number_salt_bridges: IntArray

    def __str__(self) -> str:
        """
        Writes the dataclass into a tabular string.

        Returns:
            str: 
                Output of the analysis as a tabular string.
        """
        written_output = f"{'time (ps)':<15} number\n"
        for time_point, n_salt in zip(self.time_points, self.number_salt_bridges):
            written_output += f"{time_point:<15.2f} {n_salt:<15}\n"
        return written_output

@dataclass
class HydrogenBondsOutput(BaseOutput):
    """
    Class with the outputs of a Hydrogen Bonds calculation.
    """
    time_points: FloatArray
    number_hbonds: IntArray

    def __str__(self) -> str:
        """
        Writes the dataclass into a tabular string.

        Returns:
            str: 
                Output of the analysis as a tabular string.
        """
        written_output = f"{'time (ps)':<15} number\n"
        for time_point, n_hbonds in zip(self.time_points, self.number_hbonds):
            written_output += f"{time_point:<15.2f} {n_hbonds:<15}\n"
        return written_output
