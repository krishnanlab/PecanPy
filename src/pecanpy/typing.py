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
# issue with type alias (https://stackoverflow.com/questions/62073473)
Embeddings: TypeAlias = NDArray[(Any, Any), np.float32]  # type: ignore
AdjMat: TypeAlias = NDArray[(Any, Any), Any]  # type: ignore
AdjNonZeroMat: TypeAlias = NDArray[(Any, Any), bool]  # type: ignore
Uint32Array: TypeAlias = NDArray[(Any,), np.uint32]  # type: ignore
Uint64Array: TypeAlias = NDArray[(Any,), np.uint64]  # type: ignore
Float32Array: TypeAlias = NDArray[(Any,), np.float32]  # type: ignore
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
