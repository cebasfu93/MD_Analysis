"""Module with classes that handle the outputs of the analysis functions."""
from typing import Dict, Optional
from aupt.base_io import BaseOutput
from aupt.types import FloatArray

from aupt.utilities import update_dictionary_with_key_control


class RadialDistributionFunctionsOutput(BaseOutput):
    """
    Class with the outputs of an RDF calculation.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initializer.
        """
        super().__init__(**kwargs)
        self.space: Optional[FloatArray] = None
        self.rdfs: Optional[Dict[str, FloatArray]] = None
        self.cumulative_rdfs: Optional[Dict[str, FloatArray]] = None

    def add_space(self, space: FloatArray) -> None:
        """
        Define the center of the bins for which the RDF values are valid.

        Args:
            space (FloatArray): 
                Center of the bins where the RDF values are valid.
        """
        if self.space is not None:
            raise ValueError(
                "You are trying to override the space attribute of your RDF results.")
        self.space = space

    def add_rdfs(self, rdfs: Dict[str, FloatArray], cumulative_rdfs: Dict[str, FloatArray]) -> None:
        """
        Adds a new dictionary of RDFs to the output.

        Args:
            rdfs (Dict[str, FloatArray]): 
                RDFs to add. With a label as key and the RDF as value.
            cumulative_rdfs (Dict[str, FloatArray]): 
                Cumulative RDFs to add. With a label as key and the cumulative RDF as value.

        Raises:
            ValueError: 
                If there is already data in the rdfs or cumulative_rdfs attributes.
        """
        if self.rdfs is not None or self.cumulative_rdfs is not None:
            raise ValueError(
                "You are trying to override the RDFs of the output object. "
                "Maybe you want to use update_rdfs().")
        self.rdfs = rdfs
        self.cumulative_rdfs = cumulative_rdfs

    def update_rdfs(
        self,
        rdfs: Optional[Dict[str, FloatArray]],
        cumulative_rdfs: Optional[Dict[str, FloatArray]],
        override: bool = False
    ) -> None:
        """
        Updates the RDFs or cumulative RDFs of the output.

        Args:
            rdfs (Optional[Dict[str, FloatArray]]): 
                RDFs used for the update.
            cumulative_rdfs (Optional[Dict[str, FloatArray]]): 
                Cumulative RDFs used for the update.
            override (bool, optional): 
                Whether to override or not the RDFs in the output object.

        Raises:
            ValueError: 
                If the rdfs or cumulative_rdfs haven't been initialized.
        """
        if rdfs is None and cumulative_rdfs is None:
            print("Unnecessary call to update_rdfs(). Not updating anything.")
            return
        if rdfs is not None:
            if self.rdfs is None:
                raise ValueError(
                    "There are no RDFs to update. Add new ones first.")
            self.rdfs = update_dictionary_with_key_control(
                dict1=self.rdfs, dict2=rdfs, override=override)
        if cumulative_rdfs is not None:
            if self.cumulative_rdfs is None:
                raise ValueError(
                    "There are no cumulative RDFs to update. Add new ones first.")
            self.cumulative_rdfs = update_dictionary_with_key_control(
                dict1=self.cumulative_rdfs, dict2=cumulative_rdfs, override=override)
