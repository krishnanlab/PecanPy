import unittest

import numpy as np
from pecanpy.graph import BaseGraph, AdjlstGraph, SparseGraph, DenseGraph

MAT = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
INDPTR = np.array([0, 2, 3, 4], dtype=np.uint32)
INDICES = np.array([1, 2, 0, 0], dtype=np.uint32)
DATA = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
ADJLST = [{1: 1.0, 2: 1.0}, {0: 1}, {0: 1}]
IDS = ["a", "b", "c"]
IDMAP = {"a": 0, "b": 1, "c": 2}


class TestBaseGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = BaseGraph()
        cls.g.set_ids(IDS)

    def test_set_ids(self):
        self.assertEqual(self.g.IDlst, IDS)
        self.assertEqual(self.g.IDmap, IDMAP)

    def test_properties(self):
        self.assertEqual(self.g.num_nodes, 3)
        with self.assertRaises(NotImplementedError):
            self.assertEqual(self.g.num_edges, 4)
        with self.assertRaises(NotImplementedError):
            self.assertEqual(self.g.density, 2/3)


class TestAdjlstGraph(unittest.TestCase):
    def test_from_mat(self):
        g = AdjlstGraph.from_mat(MAT, IDS)
        self.assertEqual(g._data, ADJLST)
        self.assertEqual(g.IDlst, IDS)

    def test_properties(self):
        self.g = AdjlstGraph.from_mat(MAT, IDS)
        self.assertEqual(self.g.num_nodes, 3)
        self.assertEqual(self.g.num_edges, 4)
        self.assertEqual(self.g.density, 2/3)


class TestSparseGraph(unittest.TestCase):
    def tearDown(self):
        del self.g

    def validate(self):
        self.assertTrue(np.all(self.g.indptr == INDPTR))
        self.assertTrue(np.all(self.g.indices == INDICES))
        self.assertTrue(np.all(self.g.data == DATA))
        self.assertEqual(self.g.IDlst, IDS)

    def test_from_mat(self):
        self.g = SparseGraph.from_mat(MAT, IDS)
        self.validate()

    def test_from_adjlst_graph(self):
        self.g = SparseGraph.from_adjlst_graph(AdjlstGraph.from_mat(MAT, IDS))
        self.validate()

    def test_properties(self):
        self.g = SparseGraph.from_mat(MAT, IDS)
        self.assertEqual(self.g.num_nodes, 3)
        self.assertEqual(self.g.num_edges, 4)
        self.assertEqual(self.g.density, 2/3)

class TestDenseGraph(unittest.TestCase):
    def tearDown(self):
        del self.g

    def validate(self):
        self.assertTrue(np.all(self.g.data == MAT))
        self.assertEqual(self.g.IDlst, IDS)

    def test_from_mat(self):
        self.g = DenseGraph.from_mat(MAT, IDS)
        self.validate()

    def test_from_adjlst_graph(self):
        self.g = DenseGraph.from_adjlst_graph(AdjlstGraph.from_mat(MAT, IDS))
        self.validate()

    def test_properties(self):
        self.g = DenseGraph.from_mat(MAT, IDS)
        self.assertEqual(self.g.num_nodes, 3)
        self.assertEqual(self.g.num_edges, 4)
        self.assertEqual(self.g.density, 2/3)


if __name__ == "__main__":
    unittest.main()
