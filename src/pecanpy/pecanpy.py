"""Different strategies for generating node2vec walks."""
import numpy as np
from gensim.models import Word2Vec
from numba import njit
from numba import prange
from numba.np.ufunc.parallel import _get_thread_id
from numba_progress import ProgressBar

from .graph import BaseGraph
from .rw import DenseRWGraph
from .rw import SparseRWGraph
from .typing import Embeddings
from .typing import Float32Array
from .typing import HasNbrs
from .typing import List
from .typing import MoveForward
from .typing import Optional
from .typing import Uint32Array
from .typing import Uint64Array
from .wrappers import Timer


class Base(BaseGraph):
    """Base node2vec object.

    This base object provides the skeleton for the node2vec walk algorithm,
    which consists of the ``simulate_walks`` method that generate node2vec
    random walks. In contrast to the original Python implementaion of node2vec,
    it is prallelized where each process generate walks independently.

    Note:
        The ``preprocess_transition_probs`` is required for implenetations that
        precomputes and store 2nd order transition probabilities.

    Examples:
        Generate node2vec embeddings

        >>> from pecanpy import pecanpy as node2vec
        >>>
        >>> # initialize node2vec object, similarly for SparseOTF and DenseOTF
        >>> g = node2vec.PreComp(p=0.5, q=1, workers=4, verbose=True)
        >>> # alternatively, can specify ``extend=True`` for using node2vec+
        >>>
        >>> # load graph from edgelist file
        >>> g.read_edg(path_to_edg_file, weighted=True, directed=False)
        >>> # precompute and save 2nd order transition probs (for PreComp only)
        >>> g.preprocess_transition_probs()
        >>>
        >>> # generate random walks, which could then be used to train w2v
        >>> walks = g.simulate_walks(num_walks=10, walk_length=80)
        >>>
        >>> # alternatively, generate the embeddings directly using ``embed``
        >>> emd = g.embed()

    """

    def __init__(
        self,
        p: float = 1,
        q: float = 1,
        workers: int = 1,
        verbose: bool = False,
        extend: bool = False,
        gamma: float = 0,
        random_state: Optional[int] = None,
    ):
        """Initializ node2vec base class.

        Args:
            p (float): return parameter, value less than 1 encourages returning
                back to previous vertex, and discourage for value grater than 1
                (default: 1).
            q (float): in-out parameter, value less than 1 encourages walks to
                go "outward", and value greater than 1 encourage walking within
                a localized neighborhood (default: 1)
            workers (int): number of threads to be spawned for runing node2vec
                including walk generation and word2vec embedding (default: 1)
            verbose (bool): show progress bar for walk generation.
            extend (bool): use node2vec+ extension if set to :obj:`True`
                (default: :obj:`False`).
            gamma (float): Multiplication factor for the std term of edge
                weights added to the average edge weights as the noisy edge
                threashold, only used by node2vec+ (default: 0)
            random_state (int, optional): Random seed for generating random
                walks. Note that to fully ensure reproducibility, use single
                thread (i.e., workers=1), and potentially need to set the
                Python environment variable ``PYTHONHASHSEED`` to match the
                random_state (default: :obj:`None`).

        """
        super().__init__()
        self.p = p
        self.q = q
        self.workers = workers  # TODO: not doing anything, need to fix.
        self.verbose = verbose
        self.extend = extend
        self.gamma = gamma
        self.random_state = random_state
        self._preprocessed: bool = False

    def _map_walk(self, walk_idx_ary: Uint32Array) -> List[str]:
        """Map walk from node index to node ID.

        Note:
            The last element in the ``walk_idx_ary`` encodes the effective walk
            length. Only walk indices up to the effective walk length are
            translated (mapped to node IDs).

        """
        end_idx = walk_idx_ary[-1]
        walk = [self.nodes[i] for i in walk_idx_ary[:end_idx]]
        return walk

    def simulate_walks(
        self,
        num_walks: int,
        walk_length: int,
    ) -> List[List[str]]:
        """Generate walks starting from each nodes ``num_walks`` time.

        Note:
            This is the master process that spawns worker processes, where the
            worker function ``node2vec_walks`` genearte a single random walk
            starting from a vertex of the graph.

        Args:
            num_walks (int): number of walks starting from each node.
            walks_length (int): length of walk.

        """
        self._preprocess_transition_probs()

        nodes = np.array(range(self.num_nodes), dtype=np.uint32)
        start_node_idx_ary = np.concatenate([nodes] * num_walks)
        tot_num_jobs = start_node_idx_ary.size

        random_state = self.random_state
        np.random.seed(random_state)
        np.random.shuffle(start_node_idx_ary)  # for balanced work load

        move_forward = self.get_move_forward()
        has_nbrs = self.get_has_nbrs()
        verbose = self.verbose

        # Acquire numba progress proxy for displaying the progress bar
        with ProgressBar(total=tot_num_jobs, disable=not verbose) as progress:
            walk_idx_mat = self._random_walks(
                tot_num_jobs,
                walk_length,
                random_state,
                start_node_idx_ary,
                has_nbrs,
                move_forward,
                progress,
            )

        # Map node index back to node ID
        walks = [self._map_walk(walk_idx_ary) for walk_idx_ary in walk_idx_mat]

        return walks

    @staticmethod
    @njit(parallel=True, nogil=True)
    def _random_walks(
        tot_num_jobs: int,
        walk_length: int,
        random_state: Optional[int],
        start_node_idx_ary: Uint32Array,
        has_nbrs: HasNbrs,
        move_forward: MoveForward,
        progress_proxy: ProgressBar,
    ) -> Uint32Array:
        """Simulate a random walk starting from start node."""
        # Seed the random number generator
        if random_state is not None:
            np.random.seed(random_state + _get_thread_id())

        # use the last entry of each walk index array to keep track of the
        # effective walk length
        walk_idx_mat = np.zeros((tot_num_jobs, walk_length + 2), dtype=np.uint32)
        walk_idx_mat[:, 0] = start_node_idx_ary  # initialize seeds
        walk_idx_mat[:, -1] = walk_length + 1  # set to full walk length by default

        for i in prange(tot_num_jobs):
            # initialize first step as normal random walk
            start_node_idx = walk_idx_mat[i, 0]
            if has_nbrs(start_node_idx):
                walk_idx_mat[i, 1] = move_forward(start_node_idx)
            else:
                walk_idx_mat[i, -1] = 1
                continue

            # start bias random walk
            for j in range(2, walk_length + 1):
                cur_idx = walk_idx_mat[i, j - 1]
                if has_nbrs(cur_idx):
                    prev_idx = walk_idx_mat[i, j - 2]
                    walk_idx_mat[i, j] = move_forward(cur_idx, prev_idx)
                else:
                    walk_idx_mat[i, -1] = j
                    break

            progress_proxy.update(1)

        return walk_idx_mat

    def setup_get_normalized_probs(self):
        """Transition probability computation setup.

        This is function performs necessary preprocessing of computing the
        average edge weights array, which is used later by the transition
        probability computation function ``get_extended_normalized_probs``,
        if node2vec+ is used. Otherwise, return the normal transition function
        ``get_noramlized_probs`` with a trivial placeholder for average edge
        weights array ``noise_thresholds``.

        """
        if self.extend:  # use n2v+
            get_normalized_probs = self.get_extended_normalized_probs
            noise_thresholds = self.get_noise_thresholds()
        else:  # use normal n2v
            get_normalized_probs = self.get_normalized_probs
            noise_thresholds = None
        return get_normalized_probs, noise_thresholds

    def preprocess_transition_probs(self):
        """Null default preprocess method."""
        pass

    def _preprocess_transition_probs(self):
        if not self._preprocessed:
            self.preprocess_transition_probs()
            self._preprocessed = True

    def embed(
        self,
        dim: int = 128,
        num_walks: int = 10,
        walk_length: int = 80,
        window_size: int = 10,
        epochs: int = 1,
        verbose: bool = False,
    ) -> Embeddings:
        """Generate embeddings.

        This is a shortcut function that combines ``simulate_walks`` with
        ``Word2Vec`` to generate the node2vec embedding.

        Note:
            The resulting embeddings are aligned with the graph, i.e., the
            index of embeddings is the same as that for the graph.

        Args:
            dim (int): dimension of the final embedding, default is 128
            num_walks (int): number of random walks generated using each node
                as the seed node, default is 10
            walk_length (int): length of the random walks, default is 80
            window_size (int): context window sized for training the
                ``Word2Vec`` model, default is 10
            epochs (int): number of epochs for training ``Word2Vec``, default
                is 1
            verbose (bool): print time usage for random walk generation and
                skip-gram training if set to True

        Return:
            Embeddings: The embedding matrix, each row is a node embedding
                vector. The index is the same as that for the graph.

        """
        timed_walk = Timer("generate walks", verbose)(self.simulate_walks)
        timed_w2v = Timer("train embeddings", verbose)(Word2Vec)

        walks = timed_walk(num_walks, walk_length)
        w2v = timed_w2v(
            walks,
            vector_size=dim,
            window=window_size,
            sg=1,
            min_count=0,
            workers=self.workers,
            epochs=epochs,
            seed=self.random_state,
        )

        return w2v.wv[self.nodes]


class FirstOrderUnweighted(Base, SparseRWGraph):
    """Directly sample edges for first order random walks."""

    def __init__(self, *args, **kwargs):
        """Initialize FirstOrderUnweighted mode."""
        Base.__init__(self, *args, **kwargs)

    def get_move_forward(self):
        """Wrap ``move_forward``."""
        indices = self.indices
        indptr = self.indptr

        @njit(nogil=True)
        def move_forward(cur_idx, prev_idx=None):
            start, end = indptr[cur_idx], indptr[cur_idx + 1]
            return indices[np.random.randint(start, end)]

        return move_forward


class PreCompFirstOrder(Base, SparseRWGraph):
    """Precompute transition probabilities for first order random walks."""

    def __init__(self, *args, **kwargs):
        """Initialize PreCompFirstOrder mode."""
        Base.__init__(self, *args, **kwargs)
        self.alias_j = self.alias_q = None

    def get_move_forward(self):
        """Wrap ``move_forward``."""
        indices = self.indices
        indptr = self.indptr

        alias_j = self.alias_j
        alias_q = self.alias_q

        @njit(nogil=True)
        def move_forward(cur_idx, prev_idx=None):
            start, end = indptr[cur_idx], indptr[cur_idx + 1]
            choice = alias_draw(alias_j[start:end], alias_q[start:end])

            return indices[indptr[cur_idx] + choice]

        return move_forward

    def preprocess_transition_probs(self):
        """Precompute and store first order transition probabilities."""
        data = self.data
        indices = self.indices
        indptr = self.indptr

        # Retrieve transition probability computation callback function
        get_normalized_probs = self.get_normalized_probs_first_order

        # Determine the dimensionality of the 1st order transition probs
        n_nodes = indptr.size - 1  # number of nodes
        n_probs = indptr[-1]  # total number of 1st order transition probs

        @njit(parallel=True, nogil=True)
        def compute_all_transition_probs():
            alias_j = np.zeros(n_probs, dtype=np.uint32)
            alias_q = np.zeros(n_probs, dtype=np.float32)

            for idx in range(n_nodes):
                start, end = indptr[idx], indptr[idx + 1]
                probs = get_normalized_probs(data, indices, indptr, idx)
                alias_j[start:end], alias_q[start:end] = alias_setup(probs)

            return alias_j, alias_q

        self.alias_j, self.alias_q = compute_all_transition_probs()


class PreComp(Base, SparseRWGraph):
    """Precompute transition probabilites.

    This implementation precomputes and store 2nd order transition probabilites
    first and uses read off transition probabilities during the process of
    random walk. The graph type used is ``SparseRWGraph``.

    Note:
        Need to call ``preprocess_transition_probs()`` first before generating
        walks.

    """

    def __init__(self, *args, **kwargs):
        """Initialize PreComp mode node2vec."""
        Base.__init__(self, *args, **kwargs)
        self.alias_dim: Optional[Uint32Array] = None
        self.alias_j: Optional[Uint32Array] = None
        self.alias_q: Optional[Float32Array] = None
        self.alias_indptr: Optional[Uint64Array] = None

    def get_move_forward(self):
        """Wrap ``move_forward``.

        This function returns a ``numba.njit`` compiled function that takes
        current vertex index (and the previous vertex index if available) and
        return the next vertex index by sampling from a discrete random
        distribution based on the transition probabilities that are read off
        the precomputed transition probabilities table.

        Note:
            The returned function is used by the ``simulate_walks`` method.

        """
        data = self.data
        indices = self.indices
        indptr = self.indptr
        p = self.p
        q = self.q
        get_normalized_probs = self.get_normalized_probs

        alias_j = self.alias_j
        alias_q = self.alias_q
        alias_indptr = self.alias_indptr
        alias_dim = self.alias_dim

        @njit(nogil=True)
        def move_forward(cur_idx, prev_idx=None):
            """Move to next node based on transition probabilities."""
            if prev_idx is None:
                normalized_probs = get_normalized_probs(
                    data,
                    indices,
                    indptr,
                    p,
                    q,
                    cur_idx,
                    None,
                    None,
                )
                cdf = np.cumsum(normalized_probs)
                choice = np.searchsorted(cdf, np.random.random())
            else:
                # Find index of neighbor (previous node) for reading alias
                start = indptr[cur_idx]
                end = indptr[cur_idx + 1]
                nbr_idx = np.searchsorted(indices[start:end], prev_idx)
                if indices[start + nbr_idx] != prev_idx:
                    print("FATAL ERROR! Neighbor not found.")

                dim = alias_dim[cur_idx]
                start = alias_indptr[cur_idx] + dim * nbr_idx
                end = start + dim
                choice = alias_draw(alias_j[start:end], alias_q[start:end])

            return indices[indptr[cur_idx] + choice]

        return move_forward

    def preprocess_transition_probs(self):
        """Precompute and store 2nd order transition probabilities.

        Each node contains n ** 2 number of 2nd order transition probabilities,
        where n is the number of neigbors of that specific nodes, since one can
        pick any one of its neighbors as the previous node and / or the next
        node. For each second order transition probability of a node, set up
        the alias draw table to be used during random walk.

        Note:
            Uses uint64 instaed of uint32 for tracking alias_indptr to prevent
            overflowing since the 2nd order transition probs grows much faster
            than the first order transition probs, which is the same as the
            total number of edges in the graph.

        """
        data = self.data
        indices = self.indices
        indptr = self.indptr
        p = self.p
        q = self.q

        # Retrieve transition probability computation callback function
        get_normalized_probs, noise_thresholds = self.setup_get_normalized_probs()

        # Determine the dimensionality of the 2nd order transition probs
        n_nodes = self.indptr.size - 1  # number of nodes
        n = self.indptr[1:] - self.indptr[:-1]  # number of nbrs per node
        n2 = np.power(n, 2)  # number of 2nd order trans probs per node

        # Set the dimensionality of alias probability table
        self.alias_dim = alias_dim = n
        self.alias_indptr = alias_indptr = np.zeros(self.indptr.size, dtype=np.uint64)
        alias_indptr[1:] = np.cumsum(n2)
        n_probs = alias_indptr[-1]  # total number of 2nd order transition probs

        @njit(parallel=True, nogil=True)
        def compute_all_transition_probs():
            alias_j = np.zeros(n_probs, dtype=np.uint32)
            alias_q = np.zeros(n_probs, dtype=np.float32)

            for idx in range(n_nodes):
                offset = alias_indptr[idx]
                dim = alias_dim[idx]

                nbrs = indices[indptr[idx] : indptr[idx + 1]]
                for nbr_idx in prange(n[idx]):
                    nbr = nbrs[nbr_idx]
                    probs = get_normalized_probs(
                        data,
                        indices,
                        indptr,
                        p,
                        q,
                        idx,
                        nbr,
                        noise_thresholds,
                    )

                    start = offset + dim * nbr_idx
                    end = start + dim
                    alias_j[start:end], alias_q[start:end] = alias_setup(probs)

            return alias_j, alias_q

        self.alias_j, self.alias_q = compute_all_transition_probs()


class SparseOTF(Base, SparseRWGraph):
    """Sparse graph transition on the fly.

    This implementation do *NOT* precompute transition probabilities in advance
    but instead calculate them on-the-fly during the process of random walk.
    The graph type used is ``SparseRWGraph``.

    """

    def __init__(self, *args, **kwargs):
        """Initialize PreComp mode node2vec."""
        Base.__init__(self, *args, **kwargs)

    def get_move_forward(self):
        """Wrap ``move_forward``.

        This function returns a ``numba.njit`` compiled function that takes
        current vertex index (and the previous vertex index if available) and
        return the next vertex index by sampling from a discrete random
        distribution based on the transition probabilities that are calculated
        on-the-fly.

        Note:
            The returned function is used by the ``simulate_walks`` method.

        """
        data = self.data
        indices = self.indices
        indptr = self.indptr
        p = self.p
        q = self.q

        get_normalized_probs, noise_thresholds = self.setup_get_normalized_probs()

        @njit(nogil=True)
        def move_forward(cur_idx, prev_idx=None):
            """Move to next node."""
            normalized_probs = get_normalized_probs(
                data,
                indices,
                indptr,
                p,
                q,
                cur_idx,
                prev_idx,
                noise_thresholds,
            )
            cdf = np.cumsum(normalized_probs)
            choice = np.searchsorted(cdf, np.random.random())

            return indices[indptr[cur_idx] + choice]

        return move_forward


class DenseOTF(Base, DenseRWGraph):
    """Dense graph transition on the fly.

    This implementation do *NOT* precompute transition probabilities in advance
    but instead calculate them on-the-fly during the process of random walk.
    The graph type used is ``DenseRWGraph``.

    """

    def __init__(self, *args, **kwargs):
        """Initialize DenseOTF mode node2vec."""
        Base.__init__(self, *args, **kwargs)

    def get_move_forward(self):
        """Wrap ``move_forward``.

        This function returns a ``numba.njit`` compiled function that takes
        current vertex index (and the previous vertex index if available) and
        return the next vertex index by sampling from a discrete random
        distribution based on the transition probabilities that are calculated
        on-the-fly.

        Note:
            The returned function is used by the ``simulate_walks`` method.

        """
        data = self.data
        nonzero = self.nonzero
        p = self.p
        q = self.q

        get_normalized_probs, noise_thresholds = self.setup_get_normalized_probs()

        @njit(nogil=True)
        def move_forward(cur_idx, prev_idx=None):
            """Move to next node."""
            normalized_probs = get_normalized_probs(
                data,
                nonzero,
                p,
                q,
                cur_idx,
                prev_idx,
                noise_thresholds,
            )
            cdf = np.cumsum(normalized_probs)
            choice = np.searchsorted(cdf, np.random.random())
            nbrs = np.where(nonzero[cur_idx])[0]

            return nbrs[choice]

        return move_forward


@njit(nogil=True)
def alias_setup(probs):
    """Construct alias lookup table.

    This code is modified from the blog post here:
    https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    , where you can find more details about how the method work. In general,
    the alias method improves the time complexity of sampling from a discrete
    random distribution to O(1) if the alias table is setup in advance.

    Args:
        probs (list(float32)): normalized transition probabilities array, could
            be in either list or NDArray, of float32 values.

    """
    k = probs.size
    q = np.zeros(k, dtype=np.float32)
    j = np.zeros(k, dtype=np.uint32)

    smaller = np.zeros(k, dtype=np.uint32)
    larger = np.zeros(k, dtype=np.uint32)
    smaller_ptr = 0
    larger_ptr = 0

    for kk in range(k):
        q[kk] = k * probs[kk]
        if q[kk] < 1.0:
            smaller[smaller_ptr] = kk
            smaller_ptr += 1
        else:
            larger[larger_ptr] = kk
            larger_ptr += 1

    while (smaller_ptr > 0) & (larger_ptr > 0):
        smaller_ptr -= 1
        small = smaller[smaller_ptr]
        larger_ptr -= 1
        large = larger[larger_ptr]

        j[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller[smaller_ptr] = large
            smaller_ptr += 1
        else:
            larger[larger_ptr] = large
            larger_ptr += 1

    return j, q


@njit(nogil=True)
def alias_draw(j, q):
    """Draw sample from a non-uniform discrete distribution using alias sampling."""
    k = j.size

    kk = np.random.randint(k)
    if np.random.rand() < q[kk]:
        return kk
    else:
        return j[kk]
