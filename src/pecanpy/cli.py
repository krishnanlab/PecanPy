"""Command line utility for PecanPy.

This is the command line interface for the ``pecanpy`` package.

Examples:
    Run PecanPy in command line using ``PreComp`` mode to embed the karate network::

        $ pecanpy --input demo/karate.edg --ouptut demo/karate.emb --mode PreComp

    Checkout the full list of parameters by::

        $ pecanpy --help

"""

import argparse

import numba
import numpy as np
from pecanpy import node2vec
from pecanpy.wrappers import Timer


def parse_args():
    """Parse node2vec arguments."""
    parser = argparse.ArgumentParser(
        description="Run pecanpy, a parallelized, efficient, and accelerated "
        "Python implementataion of node2vec",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        default="graph/karate.edg",
        help="Input graph (.edg or .npz) file path.",
    )

    parser.add_argument(
        "--output",
        default="emb/karate.emb",
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
        choices=["PreComp", "SparseOTF", "DenseOTF"],
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
        dest="verbose",
        action="store_true",
        help="Print out training details",
    )
    parser.set_defaults(verbose=False)

    parser.add_argument(
        "--extend",
        dest="extend",
        action="store_true",
        help="Use node2vec+ extension",
    )
    parser.set_defaults(extend=False)

    return parser.parse_args()


def check_mode(g, mode):
    """Check mode selection.

    Give recommendation to user for pecanpy mode based on graph size and density.

    """
    g_size = len(g.IDlst)  # number of nodes in graph
    if mode in ["PreComp", "SparseOTF"]:
        edge_num = sum(len(i) for i in g.data) if type(g.data) == list else g.data.size
    else:
        edge_num = g.nonzero.sum()
    g_dens = edge_num / g_size / (g_size - 1)

    if (g_dens >= 0.2) & (mode != "DenseOTF"):
        print(
            f"WARNING: network density = {g_dens:.3f} (> 0.2), recommend "
            f"DenseOTF over {mode}",
        )
    if (g_dens < 0.001) & (g_size < 10000) & (mode != "PreComp"):
        print(
            f"WARNING: network density = {g_dens:.2e} (< 0.001) with "
            f"{g_size} nodes (< 10000), recommend PreComp over {mode}",
        )
    if (g_dens >= 0.001) & (g_dens < 0.2) & (mode != "SparseOTF"):
        print(
            f"WARNING: network density = {g_dens:.3f}, recommend SparseOTF over {mode}",
        )
    if (g_dens < 0.001) & (g_size >= 10000) & (mode != "SparseOTF"):
        print(
            f"WARNING: network density = {g_dens:.3f} (< 0.001) with "
            f"{g_size} nodes (>= 10000), recommend SparseOTF over {mode}",
        )


@Timer("load Graph")
def read_graph(args):
    """Read input network to memory.

    Depending on the mode selected, reads the network either in CSR representation
    (``PreComp`` and ``SparseOTF``) or 2d numpy array (``DenseOTF``).

    """
    fp = args.input
    output = args.output
    p = args.p
    q = args.q
    workers = args.workers
    verbose = args.verbose
    weighted = args.weighted
    directed = args.directed
    extend = args.extend
    mode = args.mode
    task = args.task

    if directed and extend:
        raise NotImplementedError("Node2vec+ not implemented for directed graph yet.")

    if extend and not weighted:
        print("NOTE: node2vec+ is equivalent to node2vec for unweighted graphs.")

    if task in ["tocsr", "todense"]:  # perform conversion then save and exit
        g = node2vec.SparseGraph() if task == "tocsr" else node2vec.DenseGraph()
        g.read_edg(fp, weighted, directed)
        g.save(output)
        exit()

    pecanpy_mode = getattr(node2vec, mode, None)
    g = pecanpy_mode(p, q, workers, verbose, extend)

    read_func = g.read_npz if fp.endswith(".npz") else g.read_edg
    read_func(fp, weighted, directed)

    check_mode(g, mode)

    return g


@Timer("train embeddings")
def learn_embeddings(args, walks):
    """Learn embeddings by optimizing the Skipgram objective using SGD."""
    model = node2vec.Word2Vec(
        walks,
        vector_size=args.dimensions,
        window=args.window_size,
        min_count=0,
        sg=1,
        workers=args.workers,
        epochs=args.epochs,
    )

    output_fp = args.output
    if output_fp.endswith(".npz"):
        np.savez(output_fp, IDs=model.wv.index_to_key, data=model.wv.vectors)
    else:
        model.wv.save_word2vec_format(output_fp)


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
