"""Module with classes that handle the outputs of the temporal analysis functions."""
from dataclasses import dataclass
from typing import Dict

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
