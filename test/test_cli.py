import os
import unittest
from unittest.mock import patch
import subprocess

import numpy as np
from numba import set_num_threads
from pecanpy import cli

set_num_threads(1)


class TestCli(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        com = ["pecanpy", "--input", "../demo/karate.edg", "--output"]
        subprocess.run(com + ["../demo/karate.csr.npz", "--task", "tocsr"])
        subprocess.run(com + ["../demo/karate.dense.npz", "--task", "todense"])

    @classmethod
    def tearDownClass(cls):
        os.remove("../demo/karate.csr.npz")
        os.remove("../demo/karate.dense.npz")

    @patch("argparse._sys.argv", ["pecanpy", "--input", ""])
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
        self.execute("PreComp", "../demo/karate.edg")

    def test_sparseotf_from_edg(self):
        self.execute("SparseOTF", "../demo/karate.edg")

    def test_denseotf_from_edg(self):
        self.execute("DenseOTF", "../demo/karate.edg")

    def test_precomp_from_npz(self):
        self.execute("PreComp", "../demo/karate.csr.npz")

    def test_sparseotf_from_npz(self):
        self.execute("SparseOTF", "../demo/karate.csr.npz")

    def test_denseotf_from_npz(self):
        self.execute("DenseOTF", "../demo/karate.dense.npz")


if __name__ == "__main__":
    unittest.main()
