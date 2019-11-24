from typing import List, Union
from pyrecord import Record

Layer = Record.create_type("Layer", "weights", "activation", "bias")

Vector = List[float]
Data = Union[Vector, List[Vector]]
