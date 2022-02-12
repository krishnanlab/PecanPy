import os
import os.path as osp
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from numba import set_num_threads
from parameterized import parameterized
from pecanpy import cli

set_num_threads(1)

DATA_DIR = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, "demo"))
EDG_FP = osp.join(DATA_DIR, "karate.edg")

TMP_DATA_DIR = tempfile.mkdtemp()
CSR_FP = osp.join(TMP_DATA_DIR, "karate.csr.npz")
DENSE_FP = osp.join(TMP_DATA_DIR, "karate.dense.npz")
COM = ["pecanpy", "--input", EDG_FP, "--output"]

SETTINGS = [
    ("FirstOrderUnweighted",),
    ("PreCompFirstOrder",),
    ("PreComp",),
    ("SparseOTF",),
    ("DenseOTF",),
]


class TestCli(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(COM + [CSR_FP, "--task", "tocsr"])
        subprocess.run(COM + [DENSE_FP, "--task", "todense"])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TMP_DATA_DIR)

    @patch(
        "argparse._sys.argv",
        ["pecanpy", "--input", "", "--output", os.devnull],
    )
    def setUp(self):
        self.args = cli.parse_args()
        self.args.workers = 1
        self.args.dimensions = 8
        self.args.walk_length = 10
        self.args.num_walks = 2
        self.g = self.walks = None

    def tearDown(self):
        del self.args
        del self.g
        del self.walks

    def execute(self, mode, input_file, p=1, q=1):
        self.args.mode = mode
        self.args.input = input_file
        self.args.p = p
        self.args.q = q
        self.g = cli.read_graph(self.args)
        cli.preprocess(self.g)
        self.walks = cli.simulate_walks(self.args, self.g)
        cli.learn_embeddings(self.args, self.walks)

    def test_firstorderunweighted_catch(self):
        for p, q in (2, 1), (1, 0.1), (0.1, 0.1):
            with self.subTest(p=p, q=q):
                with self.assertRaises(ValueError):
                    self.execute("FirstOrderUnweighted", EDG_FP, p, q)

    def test_precompfirstorder_catch(self):
        for p, q in (2, 1), (1, 0.1), (0.1, 0.1):
            with self.subTest(p=p, q=q):
                with self.assertRaises(ValueError):
                    self.execute("PreCompFirstOrder", EDG_FP, p, q)

    @parameterized.expand(SETTINGS)
    def test_from_edg(self, name):
        self.execute(name, EDG_FP)

    @parameterized.expand(SETTINGS)
    def test_from_npz(self, name):
        self.execute(name, DENSE_FP if name == "DenseOTF" else CSR_FP)


if __name__ == "__main__":
    unittest.main()
