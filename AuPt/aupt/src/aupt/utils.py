"""Utility functions used here and there throughout the package."""
from typing import Any, Dict


def update_dictionary_with_key_control(
    dict1: Dict[Any, Any],
    dict2: Dict[Any, Any],
    override: bool = False
) -> Dict[Any, Any]:
    """
    Updates a dictonary with another dictionary if there are non overlapping keys 
    or if override is True.

    Args:
        dict1 (Dict[Any, Any]): 
            Base dictionary to override.
        dict2 (Dict[Any, Any]): 
            Dictionary to override with.
        override (bool, optional): 
            Whether to override the overlapping keys or not.

    Returns:
        Dict[Any, Any]: _description_
    """
    overlapping_keys = dict1.keys() & dict2.keys()
    if override is True or len(overlapping_keys) == 0:
        return dict1.update(dict2)
    raise ValueError(
        f"Could not update the dictionary because there are overlapping keys: {overlapping_keys}")
