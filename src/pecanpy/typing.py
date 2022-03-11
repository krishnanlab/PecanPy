"""Type annotations."""
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from nptyping import NDArray
from typing_extensions import TypeAlias

# Callbacks ###################################################################
HasNbrs = Callable[[np.uint32], bool]
MoveForward = Callable[..., np.uint32]

# Numpy array types ###########################################################
Embeddings: TypeAlias = NDArray[Any, np.float32]
AdjMat: TypeAlias = NDArray[[Any, Any], Any]
AdjNonZeroMat: TypeAlias = NDArray[[Any, Any], bool]
Uint32Array: TypeAlias = NDArray[[Any], np.uint32]
Float32Array: TypeAlias = NDArray[[Any], np.float32]
CSR = Tuple[Uint32Array, Uint32Array, Float32Array]

__all__ = [
    "Any",
    "Callable",
    "Dict",
    "Iterator",
    "List",
    "Tuple",
    "Optional",
    "NDArray",
    "HasNbrs",
    "MoveForward",
    "Embeddings",
    "AdjMat",
    "AdjNonZeroMat",
    "Uint32Array",
    "Float32Array",
    "CSR",
]
