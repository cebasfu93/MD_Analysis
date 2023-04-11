"""Module with classes that handle the outputs of the temporal analysis functions."""
from dataclasses import dataclass

from aupt.base_io import BaseOutput
from aupt.types import FloatArray, IntArray


@dataclass
class ContactsNumberOutput(BaseOutput):
    """
    Class with the outputs of a Contacts Number calculation.
    """
    time_points: FloatArray
    number_of_contacts: IntArray
