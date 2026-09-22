"""Authoritative five-state Cellular Automata state contract."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np


STATE_ENCODING = "ca_five_state_int8.v1"

STATE_NON_BURNABLE = np.int8(1)
STATE_NOT_YET_BURNING = np.int8(2)
STATE_IGNITED = np.int8(3)
STATE_BLAZING = np.int8(4)
STATE_EXTINGUISHED = np.int8(5)

STATE_CODES = MappingProxyType(
	{
		"non_burnable": int(STATE_NON_BURNABLE),
		"not_yet_burning": int(STATE_NOT_YET_BURNING),
		"ignited": int(STATE_IGNITED),
		"blazing": int(STATE_BLAZING),
		"extinguished": int(STATE_EXTINGUISHED),
	}
)


def validate_full_state_grid(state: object, name: str = "state") -> np.ndarray:
	"""Return a validated two-dimensional full-state ``np.int8`` grid."""
	array = np.asarray(state)
	if array.ndim != 2:
		raise ValueError(f"{name} must be a two-dimensional full CA state grid")
	if array.dtype != np.dtype(np.int8):
		raise TypeError(f"{name} must use the full np.int8 CA state encoding")
	if not np.all(np.isin(array, tuple(STATE_CODES.values()))):
		raise ValueError(f"{name} must contain only full CA state codes 1 through 5")
	return array
