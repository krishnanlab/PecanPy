import os
import os.path as osp
import shutil
import subprocess
import tempfile
import unittest
from itertools import product

from numba import set_num_threads
from parameterized import parameterized

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
PQS = [(1, 1), (2, 1), (1, 0.1), (0.1, 0.1)]


class TestReproducibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(COM + [CSR_FP, "--task", "tocsr"])
        subprocess.run(COM + [DENSE_FP, "--task", "todense"])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TMP_DATA_DIR)

    def execute(self, mode, input_file, p=1, q=1, workers=1, random_state=42):
        out_dir = osp.join(
            TMP_DATA_DIR,
            f"{mode=}_{p=}_{q=}_{workers=}_{random_state=}",
        )
        os.makedirs(out_dir, exist_ok=True)

        set_num_threads(workers)
        envvar = os.environ.copy()
        envvar["PYTHONHASHSEED"] = "0"

        out_paths = []
        for i in (1, 2):
            out_paths.append(osp.join(out_dir, f"karate_run{i}.emd"))
            command = [
                "pecanpy",
                "--input",
                input_file,
                "--output",
                out_paths[-1],
                "--mode",
                mode,
                "--p",
                str(p),
                "--q",
                str(q),
                "--workers",
                str(workers),
                "--random_state",
                str(random_state),
            ]
            subprocess.run(command, env=envvar, capture_output=True)

        path1, path2 = out_paths
        with open(path1) as f1, open(path2) as f2:
            self.assertEqual(
                f1.read(),
                f2.read(),
                f"Embeddings from two runs are not equal."
                f"\n\tRun1: {path1}\n\tRun2: {path2}",
            )

    @parameterized.expand([(s, p, q) for (s,), (p, q) in product(SETTINGS, PQS)])
    def test_repro_single_worker(self, name, p, q):
        if name in ("FirstOrderUnweighted", "PreCompFirstOrder") and (p != 1 or q != 1):
            return

        self.execute(name, EDG_FP, p=p, q=q)

    @parameterized.expand(SETTINGS)
    @unittest.expectedFailure
    def test_repro_multi_workers(self, name):
        self.execute(name, EDG_FP, workers=4)


if __name__ == "__main__":
    unittest.main()
