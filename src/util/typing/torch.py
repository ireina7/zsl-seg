from collections import namedtuple
from typing_extensions import Annotated

Tensor = 'Tensor'
DataSet = namedtuple('DataSet', 'Instance Label')
DataLoader = 'DataLoader'
