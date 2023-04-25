"""Module with classes that handle the outputs of the analysis functions."""
from dataclasses import dataclass, fields
from typing import Dict

import numpy as np

from aupt.base_io import BaseOutput
from aupt.types import FloatArray, IntArray


@dataclass
class RadialDistributionFunctionsOutput(BaseOutput):
    """
    Class with the outputs of an RDF calculation.
    """
    space: FloatArray
    rdfs: Dict[str, FloatArray]
    cumulative_rdfs: Dict[str, FloatArray]

    def __str__(self) -> str:
        """
        Writes the dataclass into a tabular string.

        Returns:
            str: 
                Output of the analysis as a tabular string.
        """
        n_fields = len(fields(self))
        labels = self.rdfs.keys()
        assert labels == self.cumulative_rdfs.keys()  # security check

        unzipped_data = np.hstack([np.vstack(
            (self.rdfs[label], self.cumulative_rdfs[label])).T
            for label in labels])
        n_data_cols = (n_fields - 1) * len(labels)  # -1 to ignore space
        written_output = f"{'# space (A)':<8} "
        for label in labels:
            written_output += f"{label:<10} {label + ' (cum)':<10} "
        written_output += "\n"
        for space_point, data in zip(self.space, unzipped_data):
            written_output += f"{space_point:<10.2f} " + \
                ("{:<10.5f} " * n_data_cols).format(*data) + "\n"
        return written_output


@dataclass
class BindingTimesOutput(BaseOutput):
    """
    Class with the outputs of a Binding Times calculation.
    """
    binding_times: FloatArray
    start_times: FloatArray
    stop_times: FloatArray
    start_frames: IntArray
    stop_frames: IntArray
    target_residue_numbers: IntArray

    def __str__(self) -> str:
        """
        Writes the dataclass into a tabular string.

        Returns:
            str: 
                Output of the analysis as a tabular string.
        """
        n_fields = len(fields(self))
        written_output = "# "
        for field in fields(self):
            written_output += f"{field.name:<20} "
        written_output += "\n"

        zipped_data = np.vstack([getattr(self, field.name)
                                for field in fields(self)]).T

        for row in zipped_data:
            written_output += ("{:<20} " * n_fields).format(*row) + "\n"
        return written_output
