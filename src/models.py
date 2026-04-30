from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]
Matrix = NDArray[np.float64]

RhsFunction = Callable[[float, Array], Array]
RhsJacobianFunction = Callable[[float, Array], Matrix]
BoundaryFunction = Callable[[Array, Array], Array]
BoundaryJacobianFunction = Callable[[Array, Array], tuple[Matrix, Matrix]]


@dataclass
class BVPProblem:
    t0: float
    t1: float
    dim: int
    rhs: RhsFunction
    rhs_jacobian: RhsJacobianFunction
    boundary_residual: BoundaryFunction
    boundary_jacobian: BoundaryJacobianFunction
    p0: Array
    num_points: int = 400
    rtol_inner: float = 1e-8
    atol_inner: float = 1e-10
