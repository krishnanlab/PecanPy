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
from pecanpy import node2vec
from pecanpy.wrappers import Timer


def parse_args():
    """Parse node2vec arguments."""
    parser = argparse.ArgumentParser(
        description="Run pecanpy, a parallelized, efficient, and accelerated Python implementataion of node2vec")

    parser.add_argument("--input", nargs="?", default="graph/karate.edgelist", help="Input graph path")

    parser.add_argument("--output", nargs="?", default="emb/karate.emb", help="Embeddings path")

    parser.add_argument(
        "--task",
        nargs="?",
        default="pecanpy",
        help="Choose task: (pecanpy, todense). Default is pecanpy")

    parser.add_argument(
        "--mode",
        nargs="?",
        default="SparseOTF",
        help="Choose mode: (PreComp, SparseOTF, DenseOTF). Default is SparseOTF")

    parser.add_argument(
        "--dimensions",
        type=int,
        default=128,
        help="Number of dimensions. Default is 128.")

    parser.add_argument(
        "--walk-length",
        type=int,
        default=80,
        help="Length of walk per source. Default is 80.")

    parser.add_argument(
        "--num-walks",
        type=int,
        default=10,
        help="Number of walks per source. Default is 10.")

    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="Context size for optimization. Default is 10. Support list of values")

    parser.add_argument("--epochs", default=1, type=int,
                        help="Number of epochs in SGD when training Word2Vec")

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers. Default is 8. Set to 0 to use all.")

    parser.add_argument("--p", type=float, default=1, help="Return hyperparameter. Default is 1.")

    parser.add_argument("--q", type=float, default=1, help="Inout hyperparameter. Default is 1.")

    parser.add_argument(
        "--weighted",
        dest="weighted",
        action="store_true",
        help="Boolean specifying (un)weighted. Default is unweighted.")
    parser.add_argument("--unweighted", dest="unweighted", action="store_false")
    parser.set_defaults(weighted=False)

    parser.add_argument(
        "--directed",
        dest="directed",
        action="store_true",
        help="Graph is (un)directed. Default is undirected.")
    parser.add_argument("--undirected", dest="undirected", action="store_false")
    parser.set_defaults(directed=False)

    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Print out training details")
    parser.set_defaults(verbose=False)

    parser.add_argument(
        "--extend",
        dest="extend",
        action="store_true",
        help="Use node2vec+ extension")
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
        print(f"WARNING: network density = {g_dens:.3f} (> 0.2), recommend DenseOTF over {mode}")
    if (g_dens < 0.001) & (g_size < 10000) & (mode != "PreComp"):
        print(f"WARNING: network density = {g_dens:.2e} (< 0.001) with "
              f"{g_size} nodes (< 10000), recommend PreComp over {mode}")
    if (g_dens >= 0.001) & (g_dens < 0.2) & (mode != "SparseOTF"):
        print(f"WARNING: network density = {g_dens:.3f}, recommend SparseOTF over {mode}")
    if (g_dens < 0.001) & (g_size >= 10000) & (mode != "SparseOTF"):
        print(f"WARNING: network density = {g_dens:.3f} (< 0.001) with "
              f"{g_size} nodes (>= 10000), recommend SparseOTF over {mode}")


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

    if task == "todense":
        g = node2vec.DenseGraph()
        g.read_edg(fp, weighted, directed)
        g.save(output)
        exit()
    elif task != "pecanpy":
        raise ValueError(f"Unknown task: {repr(task)}")

    if mode == "PreComp":
        g = node2vec.PreComp(p, q, workers, verbose, extend)
        g.read_edg(fp, weighted, directed)
    elif mode == "SparseOTF":
        g = node2vec.SparseOTF(p, q, workers, verbose, extend)
        g.read_edg(fp, weighted, directed)
    elif mode == "DenseOTF":
        g = node2vec.DenseOTF(p, q, workers, verbose, extend)
        if fp.endswith(".npz"):
            g.read_npz(fp, weighted, directed)
        else:
            g.read_edg(fp, weighted, directed)
    else:
        raise ValueError(f"Unkown mode: {repr(mode)}")

    check_mode(g, mode)
    if extend and not weighted:
        print("WARNING: node2vec+ is equivalent to node2vec for unweighted graphs.")

    return g


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
    model.wv.save_word2vec_format(args.output)


def main():
    """Pipeline for representational learning for all nodes in a graph."""
    args = parse_args()

    if args.directed and args.extend:
        raise NotImplementedError("Node2vec+ not implemented for directed graph yet.")

    @Timer("load graph", True)
    def timed_read_graph():
        return read_graph(args)

    @Timer("pre-compute transition probabilities", True)
    def timed_preprocess():
        g.preprocess_transition_probs()

    @Timer("generate walks", True)
    def timed_walk():
        return g.simulate_walks(args.num_walks, args.walk_length)

    @Timer("train embeddings", True)
    def timed_emb():
        learn_embeddings(args=args, walks=walks)

    if args.workers == 0:
        args.workers = numba.config.NUMBA_DEFAULT_NUM_THREADS
    numba.set_num_threads(args.workers)

    g = timed_read_graph()
    timed_preprocess()
    walks = timed_walk()
    g = None
    timed_emb()


if __name__ == "__main__":
    main()
