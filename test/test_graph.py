import os
import shutil
import tempfile
import unittest

import numpy as np
from pecanpy.graph import AdjlstGraph
from pecanpy.graph import BaseGraph
from pecanpy.graph import DenseGraph
from pecanpy.graph import SparseGraph

MAT = np.array(
    [
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 0],
    ],
    dtype=float,
)
INDPTR = np.array([0, 2, 3, 4], dtype=np.uint32)
INDICES = np.array([1, 2, 0, 0], dtype=np.uint32)
DATA = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
ADJLST = [
    {1: 1.0, 2: 1.0},
    {0: 1.0},
    {0: 1.0},
]
IDS = ["a", "b", "c"]
IDMAP = {"a": 0, "b": 1, "c": 2}

# Test to make sure the IDs are loaded in order even in the case when the
# order in which the node ID appears (from edges) is not ordered correctly.
MAT2 = np.array(
    [
        [0, 1, 0, 0, 0],
        [1, 0, 1, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0],
    ],
    dtype=float,
)
INDPTR2 = np.array([0, 1, 4, 5, 7, 8], dtype=np.uint32)
INDICES2 = np.array([1, 0, 2, 3, 1, 1, 4, 3], dtype=np.uint32)
DATA2 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
ADJLST2 = [
    {1: 1.0},
    {0: 1.0, 2: 1.0, 3: 1.0},
    {1: 1.0},
    {1: 1.0, 4: 1.0},
    {3: 1.0},
]
IDS2 = ["a", "b", "c", "d", "e"]
IDMAP2 = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}

# Test asymmetric directed graph loading with node that has no out-going edge
MAT3 = np.array(
    [
        [0, 1, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 1, 1, 0],
    ],
)
INDPTR3 = np.array([0, 1, 3, 3, 5], dtype=np.uint32)
INDICES3 = np.array([1, 0, 3, 1, 2], dtype=np.uint32)
DATA3 = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
ADJLST3 = [
    {1: 1.0},
    {0: 1.0, 3: 1.0},
    {},
    {1: 1.0, 2: 1.0},
]
IDS3 = ["a", "b", "c", "d"]
IDMAP3 = {"a": 0, "b": 1, "c": 2, "d": 3}


class TestBaseGraph(unittest.TestCase):
    def setUp(self):
        self.g = BaseGraph()
        self.g.set_node_ids(IDS)

    def test_set_node_ids(self):
        self.assertEqual(self.g.nodes, IDS)
        self.assertEqual(self.g._node_idmap, IDMAP)

    def test_properties(self):
        self.assertEqual(self.g.num_nodes, 3)
        with self.assertRaises(NotImplementedError):
            self.assertEqual(self.g.num_edges, 4)
        with self.assertRaises(NotImplementedError):
            self.assertEqual(self.g.density, 2 / 3)


class TestAdjlstGraph(unittest.TestCase):
    def setUp(self):
        self.g1 = AdjlstGraph.from_mat(MAT, IDS)
        self.g2 = AdjlstGraph.from_mat(MAT2, IDS2)
        self.g3 = AdjlstGraph.from_mat(MAT3, IDS3)

    def tearDown(self):
        del self.g1
        del self.g2
        del self.g3

    def test_from_mat(self):
        self.assertEqual(self.g1._data, ADJLST)
        self.assertEqual(self.g1.nodes, IDS)

        self.assertEqual(self.g2._data, ADJLST2)
        self.assertEqual(self.g2.nodes, IDS2)

        self.assertEqual(self.g3._data, ADJLST3)
        self.assertEqual(self.g3.nodes, IDS3)

    def test_properties(self):
        self.assertEqual(self.g1.num_nodes, 3)
        self.assertEqual(self.g1.num_edges, 4)
        self.assertEqual(self.g1.density, 2 / 3)

        self.assertEqual(self.g2.num_nodes, 5)
        self.assertEqual(self.g2.num_edges, 8)
        self.assertEqual(self.g2.density, 2 / 5)

        self.assertEqual(self.g3.num_nodes, 4)
        self.assertEqual(self.g3.num_edges, 5)
        self.assertEqual(self.g3.density, 5 / 12)

    def test_edges(self):
        self.assertEqual(
            list(self.g1.edges),
            [
                (0, 1, 1),
                (0, 2, 1),
                (1, 0, 1),
                (2, 0, 1),
            ],
        )

        self.assertEqual(
            list(self.g2.edges),
            [
                (0, 1, 1),
                (1, 0, 1),
                (1, 2, 1),
                (1, 3, 1),
                (2, 1, 1),
                (3, 1, 1),
                (3, 4, 1),
                (4, 3, 1),
            ],
        )

    def test_save(self):
        expected_results = {
            (False, "\t"): [
                "a\tb\t1.0\n",
                "a\tc\t1.0\n",
                "b\ta\t1.0\n",
                "c\ta\t1.0\n",
            ],
            (True, "\t"): [
                "a\tb\n",
                "a\tc\n",
                "b\ta\n",
                "c\ta\n",
            ],
            (False, ","): [
                "a,b,1.0\n",
                "a,c,1.0\n",
                "b,a,1.0\n",
                "c,a,1.0\n",
            ],
            (True, ","): [
                "a,b\n",
                "a,c\n",
                "b,a\n",
                "c,a\n",
            ],
        }

        tmpdir = tempfile.mkdtemp()
        tmppath = os.path.join(tmpdir, "test.edg")

        for unweighted in True, False:
            for delimiter in ["\t", ","]:
                self.g1.save(
                    tmppath,
                    unweighted=unweighted,
                    delimiter=delimiter,
                )

                with open(tmppath, "r") as f:
                    expected_result = expected_results[(unweighted, delimiter)]
                    for line, expected_line in zip(f, expected_result):
                        self.assertEqual(line, expected_line)

        shutil.rmtree(tmpdir)


class TestSparseGraph(unittest.TestCase):
    def tearDown(self):
        del self.g1
        del self.g2
        del self.g3

    def validate(self):
        self.assertTrue(np.all(self.g1.indptr == INDPTR))
        self.assertTrue(np.all(self.g1.indices == INDICES))
        self.assertTrue(np.all(self.g1.data == DATA))
        self.assertEqual(self.g1.nodes, IDS)
        self.assertEqual(self.g1.num_nodes, 3)
        self.assertEqual(self.g1.num_edges, 4)
        self.assertEqual(self.g1.density, 2 / 3)

        self.assertTrue(np.all(self.g2.indptr == INDPTR2))
        self.assertTrue(np.all(self.g2.indices == INDICES2))
        self.assertTrue(np.all(self.g2.data == DATA2))
        self.assertEqual(self.g2.nodes, IDS2)
        self.assertEqual(self.g2.num_nodes, 5)
        self.assertEqual(self.g2.num_edges, 8)
        self.assertEqual(self.g2.density, 2 / 5)

        self.assertTrue(np.all(self.g3.indptr == INDPTR3))
        self.assertTrue(np.all(self.g3.indices == INDICES3))
        self.assertTrue(np.all(self.g3.data == DATA3))
        self.assertEqual(self.g3.nodes, IDS3)
        self.assertEqual(self.g3.num_nodes, 4)
        self.assertEqual(self.g3.num_edges, 5)
        self.assertEqual(self.g3.density, 5 / 12)

    def test_from_mat(self):
        self.g1 = SparseGraph.from_mat(MAT, IDS)
        self.g2 = SparseGraph.from_mat(MAT2, IDS2)
        self.g3 = SparseGraph.from_mat(MAT3, IDS3)
        self.validate()

    def test_from_adjlst_graph(self):
        self.g1 = SparseGraph.from_adjlst_graph(AdjlstGraph.from_mat(MAT, IDS))
        self.g2 = SparseGraph.from_adjlst_graph(AdjlstGraph.from_mat(MAT2, IDS2))
        self.g3 = SparseGraph.from_adjlst_graph(AdjlstGraph.from_mat(MAT3, IDS3))
        self.validate()


class TestDenseGraph(unittest.TestCase):
    def tearDown(self):
        del self.g1
        del self.g2

    def validate(self):
        self.assertTrue(np.all(self.g1.data == MAT))
        self.assertEqual(self.g1.nodes, IDS)
        self.assertEqual(self.g1.num_nodes, 3)
        self.assertEqual(self.g1.num_edges, 4)
        self.assertEqual(self.g1.density, 2 / 3)

        self.assertTrue(np.all(self.g2.data == MAT2))
        self.assertEqual(self.g2.nodes, IDS2)
        self.assertEqual(self.g2.num_nodes, 5)
        self.assertEqual(self.g2.num_edges, 8)
        self.assertEqual(self.g2.density, 2 / 5)

        self.assertTrue(np.all(self.g3.data == MAT3))
        self.assertEqual(self.g3.nodes, IDS3)
        self.assertEqual(self.g3.num_nodes, 4)
        self.assertEqual(self.g3.num_edges, 5)
        self.assertEqual(self.g3.density, 5 / 12)

    def test_from_mat(self):
        self.g1 = DenseGraph.from_mat(MAT, IDS)
        self.g2 = DenseGraph.from_mat(MAT2, IDS2)
        self.g3 = DenseGraph.from_mat(MAT3, IDS3)
        self.validate()

    def test_from_adjlst_graph(self):
        self.g1 = DenseGraph.from_adjlst_graph(AdjlstGraph.from_mat(MAT, IDS))
        self.g2 = DenseGraph.from_adjlst_graph(AdjlstGraph.from_mat(MAT2, IDS2))
        self.g3 = DenseGraph.from_adjlst_graph(AdjlstGraph.from_mat(MAT3, IDS3))
        self.validate()


if __name__ == "__main__":
    unittest.main()
