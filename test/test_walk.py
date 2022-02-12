import os.path as osp
import unittest

import numpy as np
from numba import set_num_threads
from parameterized import parameterized
from pecanpy import pecanpy

set_num_threads(1)

MAT = np.array(
    [
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 1],
        [0, 0, 1, 0, 1],
        [0, 0, 1, 1, 0],
    ],
)
IDS = ["a", "b", "c", "d", "e"]

WALKS = {
    "FirstOrderUnweighted": [
        ["c", "b", "c", "d"],
        ["d", "c", "d", "e"],
        ["e", "d", "c", "b"],
        ["e", "d", "c", "b"],
        ["b", "a", "b", "a"],
        ["b", "a", "b", "c"],
        ["c", "e", "d", "e"],
        ["d", "c", "b", "c"],
        ["a", "b", "c", "d"],
        ["a", "b", "c", "b"],
    ],
    "PreCompFirstOrder": [
        ["c", "d", "e", "d"],
        ["d", "c", "d", "e"],
        ["e", "d", "c", "e"],
        ["e", "d", "e", "c"],
        ["b", "c", "e", "c"],
        ["b", "c", "d", "c"],
        ["c", "d", "e", "d"],
        ["d", "c", "e", "d"],
        ["a", "b", "a", "b"],
        ["a", "b", "c", "e"],
    ],
    "PreComp": [
        ["c", "d", "e", "d"],
        ["d", "c", "d", "e"],
        ["e", "d", "c", "e"],
        ["e", "d", "e", "c"],
        ["b", "c", "e", "c"],
        ["b", "c", "d", "c"],
        ["c", "d", "e", "d"],
        ["d", "c", "e", "d"],
        ["a", "b", "a", "b"],
        ["a", "b", "c", "e"],
    ],
    "SparseOTF": [
        ["c", "d", "e", "d"],
        ["d", "e", "c", "d"],
        ["e", "c", "e", "d"],
        ["e", "c", "e", "d"],
        ["b", "c", "e", "c"],
        ["b", "a", "b", "c"],
        ["c", "e", "d", "e"],
        ["d", "e", "c", "e"],
        ["a", "b", "c", "b"],
        ["a", "b", "c", "d"],
    ],
    "DenseOTF": [
        ["c", "d", "e", "d"],
        ["d", "e", "c", "d"],
        ["e", "c", "e", "d"],
        ["e", "c", "e", "d"],
        ["b", "c", "e", "c"],
        ["b", "a", "b", "c"],
        ["c", "e", "d", "e"],
        ["d", "e", "c", "e"],
        ["a", "b", "c", "b"],
        ["a", "b", "c", "d"],
    ],
}


class TestWalk(unittest.TestCase):
    @parameterized.expand(
        [
            ("FirstOrderUnweighted", pecanpy.FirstOrderUnweighted),
            ("PreCompFirstOrder", pecanpy.PreComp),
            ("PreComp", pecanpy.PreComp),
            ("SparseOTF", pecanpy.SparseOTF),
            ("DenseOTF", pecanpy.DenseOTF),
        ],
    )
    def test_first_order_unweighted(self, name, mode):
        graph = mode.from_mat(MAT, IDS, p=1, q=1, random_state=0)
        walks = graph.simulate_walks(2, 3)
        self.assertEqual(walks, WALKS[name])
        print(walks)


if __name__ == "__main__":
    unittest.main()
