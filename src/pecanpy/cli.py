"""Command line utility for PecanPy.

This is the command line interface for the ``pecanpy`` package.

Examples:
    Run PecanPy in command line using ``PreComp`` mode to embed the karate network::

        $ pecanpy --input demo/karate.edg --ouptut demo/karate.emb --mode PreComp

    Checkout the full list of parameters by::

        $ pecanpy --help

"""
import argparse
import warnings

import numba
import numpy as np
from gensim.models import Word2Vec

from . import graph
from . import pecanpy
from .wrappers import Timer


def parse_args():
    """Parse node2vec arguments."""
    parser = argparse.ArgumentParser(
        description="Run pecanpy, a parallelized, efficient, and accelerated "
        "Python implementataion of node2vec",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input graph (.edg or .npz) file path.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output embeddings file path. Save as .npz file if the specified "
        "file path ends with .npz, otherwise save as a text file using the "
        "gensim save_word2vec_format method.",
    )

    parser.add_argument(
        "--task",
        default="pecanpy",
        choices=["pecanpy", "tocsr", "todense"],
        help="Task to be performed.",
    )

    parser.add_argument(
        "--mode",
        default="SparseOTF",
        choices=[
            "DenseOTF",
            "FirstOrderUnweighted",
            "PreComp",
            "PreCompFirstOrder",
            "SparseOTF",
        ],
        help="PecanPy execution mode.",
    )

    parser.add_argument(
        "--dimensions",
        type=int,
        default=128,
        help="Number of dimensions.",
    )

    parser.add_argument(
        "--walk-length",
        type=int,
        default=80,
        help="Length of walk per source.",
    )

    parser.add_argument(
        "--num-walks",
        type=int,
        default=10,
        help="Number of walks per source.",
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="Context size for optimization.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs in SGD when training Word2Vec",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers (0 to use all available threads).",
    )

    parser.add_argument(
        "--p",
        type=float,
        default=1,
        help="Return hyperparameter.",
    )

    parser.add_argument(
        "--q",
        type=float,
        default=1,
        help="Inout hyperparameter.",
    )

    parser.add_argument(
        "--weighted",
        action="store_true",
        help="Boolean specifying (un)weighted.",
    )

    parser.add_argument(
        "--directed",
        action="store_true",
        help="Graph is (un)directed.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print out training details",
    )

    parser.add_argument(
        "--extend",
        action="store_true",
        help="Use node2vec+ extension",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0,
        help="Noisy edge threshold parameter.",
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=None,
        help="Random seed for generating random walks.",
    )

    parser.add_argument(
        "--delimiter",
        type=str,
        default="\t",
        help="Delimiter used bewteen node IDs.",
    )

    parser.add_argument(
        "--implicit_ids",
        action="store_true",
        help="If set, use canonical node ordering for the node IDs.",
    )

    return parser.parse_args()


def check_mode(g, args):
    """Check mode selection.

    Give recommendation to user for pecanpy mode based on graph size and density.

    """
    mode = args.mode
    weighted = args.weighted
    p = args.p
    q = args.q

    # Check unweighted first order random walk usage
    if mode == "FirstOrderUnweighted":
        if not p == q == 1 or weighted:
            raise ValueError(
                f"FirstOrderUnweighted only works when weighted = False and "
                f"p = q = 1, got {weighted=}, {p=}, {q=}",
            )
        return

    if mode != "FirstOrderUnweighted" and p == q == 1 and not weighted:
        warnings.warn(
            "When p = 1 and q = 1 with unweighted graph, it is highly "
            f"recommended to use FirstOrderUnweighted over {mode} (current "
            "selection). The runtime could be improved greatly with improved  "
            "memory usage.",
        )
        return

    # Check first order random walk usage
    if mode == "PreCompFirstOrder":
        if not p == q == 1:
            raise ValueError(
                f"PreCompFirstOrder only works when p = q = 1, got {p=}, {q=}",
            )
        return

    if mode != "PreCompFirstOrder" and p == 1 == q:
        warnings.warn(
            "When p = 1 and q = 1, it is highly recommended to use "
            f"PreCompFirstOrder over {mode} (current selection). The runtime "
            "could be improved greatly with low memory usage.",
        )
        return

    # Check network density and recommend appropriate mode
    g_size = g.num_nodes
    g_dens = g.density
    if (g_dens >= 0.2) & (mode != "DenseOTF"):
        warnings.warn(
            f"Network density = {g_dens:.3f} (> 0.2), it is recommended to use "
            f"DenseOTF over {mode} (current selection)",
        )
    if (g_dens < 0.001) & (g_size < 10000) & (mode != "PreComp"):
        warnings.warn(
            f"Network density = {g_dens:.2e} (< 0.001) with {g_size} nodes "
            f"(< 10000), it is recommended to use PreComp over {mode} (current "
            "selection)",
        )
    if (g_dens >= 0.001) & (g_dens < 0.2) & (mode != "SparseOTF"):
        warnings.warn(
            f"Network density = {g_dens:.3f}, it is recommended to use "
            f"SparseOTF over {mode} (current selection)",
        )
    if (g_dens < 0.001) & (g_size >= 10000) & (mode != "SparseOTF"):
        warnings.warn(
            f"Network density = {g_dens:.3f} (< 0.001) with {g_size} nodes "
            f"(>= 10000), it is recommended to use SparseOTF over {mode} "
            "(current selection)",
        )


@Timer("load Graph")
def read_graph(args):
    """Read input network to memory.

    Depending on the mode selected, reads the network either in CSR
    representation (``PreComp`` and ``SparseOTF``) or 2d numpy array
    (``DenseOTF``).

    """
    path = args.input
    output = args.output
    p = args.p
    q = args.q
    workers = args.workers
    verbose = args.verbose
    weighted = args.weighted
    directed = args.directed
    extend = args.extend
    gamma = args.gamma
    random_state = args.random_state
    mode = args.mode
    task = args.task
    delimiter = args.delimiter
    implicit_ids = args.implicit_ids

    if directed and extend:
        raise NotImplementedError("Node2vec+ not implemented for directed graph yet.")

    if extend and not weighted:
        print("NOTE: node2vec+ is equivalent to node2vec for unweighted graphs.")

    if task in ["tocsr", "todense"]:  # perform conversion then save and exit
        g = graph.SparseGraph() if task == "tocsr" else graph.DenseGraph()
        g.read_edg(path, weighted, directed, delimiter)
        g.save(output)
        exit()

    pecanpy_mode = getattr(pecanpy, mode, None)
    g = pecanpy_mode(p, q, workers, verbose, extend, gamma, random_state)

    if path.endswith(".npz"):
        g.read_npz(path, weighted, implicit_ids=implicit_ids)
    else:
        g.read_edg(path, weighted, directed, delimiter)

    check_mode(g, args)

    return g


@Timer("train embeddings")
def learn_embeddings(args, walks):
    """Learn embeddings by optimizing the Skipgram objective using SGD."""
    model = Word2Vec(
        walks,
        vector_size=args.dimensions,
        window=args.window_size,
        min_count=0,
        sg=1,
        workers=args.workers,
        epochs=args.epochs,
        seed=args.random_state,
    )

    output_path = args.output
    if output_path.endswith(".npz"):
        np.savez(output_path, IDs=model.wv.index_to_key, data=model.wv.vectors)
    else:
        model.wv.save_word2vec_format(output_path)


@Timer("pre-compute transition probabilities")
def preprocess(g):
    """Preprocessing transition probabilities with timer."""
    g.preprocess_transition_probs()


@Timer("generate walks")
def simulate_walks(args, g):
    """Simulate random walks with timer."""
    return g.simulate_walks(args.num_walks, args.walk_length)


def main():
    """Pipeline for representational learning for all nodes in a graph."""
    args = parse_args()

    if args.workers == 0:
        args.workers = numba.config.NUMBA_DEFAULT_NUM_THREADS
    numba.set_num_threads(args.workers)

    g = read_graph(args)
    preprocess(g)
    walks = simulate_walks(args, g)
    learn_embeddings(args, walks)


if __name__ == "__main__":
    main()
