"""Type annotations."""
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

from nptyping import Bool
from nptyping import Float32
from nptyping import NDArray
from nptyping import Shape
from nptyping import UInt32
from nptyping import UInt64
from typing_extensions import TypeAlias

# Callbacks ###################################################################
HasNbrs = Callable[[UInt32], bool]
MoveForward = Callable[..., UInt32]

# Numpy array types ###########################################################
# issue with type alias (https://stackoverflow.com/questions/62073473)
Embeddings: TypeAlias = NDArray[Shape["*, *"], Float32]
AdjMat: TypeAlias = NDArray[Shape["*, *"], Any]
AdjNonZeroMat: TypeAlias = NDArray[Shape["*, *"], Bool]
Uint32Array: TypeAlias = NDArray[Shape["*"], UInt32]
Uint64Array: TypeAlias = NDArray[Shape["*"], UInt64]
Float32Array: TypeAlias = NDArray[Shape["*"], Float32]
CSR = Tuple[Uint32Array, Uint32Array, Float32Array]

__all__ = [
    "AdjMat",
    "AdjNonZeroMat",
    "Any",
    "CSR",
    "Callable",
    "Dict",
    "Embeddings",
    "Float32Array",
    "HasNbrs",
    "Iterator",
    "List",
    "MoveForward",
    "NDArray",
    "Optional",
    "Sequence",
    "Tuple",
    "Uint32Array",
]
