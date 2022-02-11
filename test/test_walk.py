import unittest
import os.path as osp

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

FIRST_ORDER_WALK_SEED0 = [
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
]


class TestWalk(unittest.TestCase):
    @parameterized.expand(
        [
            (pecanpy.FirstOrderUnweighted,),
        ],
    )
    def test_first_order_unweighted(self, mode):
        print(mode)
        graph = mode.from_mat(MAT, IDS, p=1, q=1, random_state=0)
        walks = graph.simulate_walks(2, 3)
        self.assertEqual(walks, FIRST_ORDER_WALK_SEED0)
        print(walks)


if __name__ == "__main__":
    unittest.main()
