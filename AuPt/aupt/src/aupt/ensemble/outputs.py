"""Module with classes that handle the outputs of the analysis functions."""
from dataclasses import dataclass
from typing import Dict
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


@ dataclass
class BindingTimesOutput(BaseOutput):
    """
    Class with the outputs of a Binding Times calculation.
    """
    binding_times: FloatArray
    start_times: FloatArray
    stop_times: FloatArray
    target_residue_numbers: IntArray
