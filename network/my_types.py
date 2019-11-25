from typing import List, Union
from pyrecord import Record

Layer = Record.create_type("Layer", "weights", "activation", "bias")
Landscape = Record.create_type("Landscape", "position", "value")

Vector = List[float]
Data = Union[Vector, List[Vector]]
