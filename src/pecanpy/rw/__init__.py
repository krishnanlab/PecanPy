"""Graph objects equipped with random walk transition functions."""
from .dense_rw import DenseRWGraph
from .sparse_rw import SparseRWGraph

__all__ = ["DenseRWGraph", "SparseRWGraph"]
