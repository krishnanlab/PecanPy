import os.path as osp
import unittest

from numba import set_num_threads
from parameterized import parameterized
from pecanpy import graph
from pecanpy import pecanpy

set_num_threads(1)

DATA_DIR = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, "demo"))
EDG_FP = osp.join(DATA_DIR, "karate.edg")
SETTINGS = [
    ("SparseOTF", pecanpy.SparseOTF),
    ("DenseOTF", pecanpy.DenseOTF),
    ("PreComp", pecanpy.PreComp),
    ("PreCompFirstOrder", pecanpy.PreCompFirstOrder),
    ("FirstOrderUnweighted", pecanpy.FirstOrderUnweighted),
]


class TestPecanPy(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        g = graph.DenseGraph()
        g.read_edg(EDG_FP, weighted=False, directed=False)
        self.mat = g.data
        self.ids = g.nodes

    @parameterized.expand(SETTINGS)
    def test_from_mat(self, name, mode):
        with self.subTest(name):
            g = mode.from_mat(self.mat, self.ids, p=1, q=1)
            g.embed()

    @parameterized.expand(SETTINGS)
    def test_from_edg(self, name, mode):
        with self.subTest(name):
            g = mode(p=1, q=1)
            g.read_edg(EDG_FP, weighted=False, directed=False)
            g.embed()


if __name__ == "__main__":
    unittest.main()
