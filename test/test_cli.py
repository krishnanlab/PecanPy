import os
import os.path as op
import unittest
from unittest.mock import patch
import subprocess

import numpy as np
from numba import set_num_threads
from pecanpy import cli

set_num_threads(1)

DATA_DIR = op.abspath(op.join(__file__, op.pardir, op.pardir, 'demo'))
EDG_FP = op.join(DATA_DIR, "karate.edg")
CSR_FP = op.join(DATA_DIR, "karate.csr.npz")
DENSE_FP = op.join(DATA_DIR, "karate.dense.npz")
COM = ["pecanpy", "--input", EDG_FP, "--output"]


class TestCli(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(COM + [CSR_FP, "--task", "tocsr"])
        subprocess.run(COM + [DENSE_FP, "--task", "todense"])

    @classmethod
    def tearDownClass(cls):
        os.remove(CSR_FP)
        os.remove(DENSE_FP)

    @patch(
        "argparse._sys.argv",
        ["pecanpy", "--input", "", "--output", "/dev/null"]
    )
    def setUp(self):
        self.args = cli.parse_args()
        self.args.workers = 1
        self.args.dimensions = 8
        self.args.walk_length = 10
        self.args.num_walks = 2

    def tearDown(self):
        del self.args
        del self.g
        del self.walks

    def execute(self, mode, input_file):
        self.args.mode = mode
        self.args.input = input_file
        self.g = cli.read_graph(self.args)
        cli.preprocess(self.g)
        self.walks = cli.simulate_walks(self.args, self.g)
        cli.learn_embeddings(self.args, self.walks)

    def test_precomp_from_edg(self):
        self.execute("PreComp", EDG_FP)

    def test_sparseotf_from_edg(self):
        self.execute("SparseOTF", EDG_FP)

    def test_denseotf_from_edg(self):
        self.execute("DenseOTF", EDG_FP)

    def test_precomp_from_npz(self):
        self.execute("PreComp", CSR_FP)

    def test_sparseotf_from_npz(self):
        self.execute("SparseOTF", CSR_FP)

    def test_denseotf_from_npz(self):
        self.execute("DenseOTF", DENSE_FP)


if __name__ == "__main__":
    unittest.main()
