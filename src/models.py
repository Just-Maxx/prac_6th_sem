from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]
RhsFunction = Callable[[float, Array], Array]
BoundaryFunction = Callable[[Array, Array], Array]


@dataclass
class BVPProblem:
    t0: float
    t1: float
    dim: int
    rhs: RhsFunction
    boundary_residual: BoundaryFunction
    p0: Array
    num_points: int = 400
    rtol_inner: float = 1e-8
    atol_inner: float = 1e-10
