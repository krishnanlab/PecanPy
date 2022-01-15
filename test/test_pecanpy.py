import os.path as op
import unittest

from numba import set_num_threads
from pecanpy import graph
from pecanpy import pecanpy

set_num_threads(1)

DATA_DIR = op.abspath(op.join(__file__, op.pardir, op.pardir, "demo"))
EDG_FP = op.join(DATA_DIR, "karate.edg")


class TestPecanPyFromMat(unittest.TestCase):
    def setUp(self):
        g = graph.DenseGraph()
        g.read_edg(EDG_FP, weighted=False, directed=False)
        self.mat = g.data
        self.ids = g.IDlst
        self.kwargs = {"p": 1, "q": 1, "workers": 1}

    def test_sparseotf_from_mat(self):
        g = pecanpy.SparseOTF.from_mat(self.mat, self.ids, **self.kwargs)
        g.embed()

    def test_denseotf_from_mat(self):
        g = pecanpy.DenseOTF.from_mat(self.mat, self.ids, **self.kwargs)
        g.embed()

    def test_precomp_from_mat(self):
        g = pecanpy.PreComp.from_mat(self.mat, self.ids, **self.kwargs)
        g.preprocess_transition_probs()
        g.embed()

    def test_precompfirtorder_from_mat(self):
        g = pecanpy.PreCompFirstOrder.from_mat(self.mat, self.ids, **self.kwargs)
        g.preprocess_transition_probs()
        g.embed()

    def test_firtorderunweighted_from_mat(self):
        g = pecanpy.FirstOrderUnweighted.from_mat(self.mat, self.ids, **self.kwargs)
        g.embed()


class TestPecanPyFromEdg(unittest.TestCase):
    def test_sparseotf_from_edg(self):
        g = pecanpy.SparseOTF(1, 1, 1)
        g.read_edg(EDG_FP, weighted=False, directed=False)
        g.embed()

    def test_denseotf_from_edg(self):
        g = pecanpy.DenseOTF(1, 1, 1)
        g.read_edg(EDG_FP, weighted=False, directed=False)
        g.embed()

    def test_precomp_from_edg(self):
        g = pecanpy.PreComp(1, 1, 1)
        g.read_edg(EDG_FP, weighted=False, directed=False)
        g.preprocess_transition_probs()
        g.embed()

    def test_precompfirstorder_from_edg(self):
        g = pecanpy.PreCompFirstOrder(1, 1, 1)
        g.read_edg(EDG_FP, weighted=False, directed=False)
        g.preprocess_transition_probs()
        g.embed()

    def test_firstorderunweighted_from_edg(self):
        g = pecanpy.FirstOrderUnweighted(1, 1, 1)
        g.read_edg(EDG_FP, weighted=False, directed=False)
        g.embed()


if __name__ == "__main__":
    unittest.main()
