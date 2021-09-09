"""Different strategies for generating node2vec walks."""

import numpy as np
from gensim.models import Word2Vec
from numba import get_num_threads, jit, prange
from numba.np.ufunc.parallel import _get_thread_id
from pecanpy.graph import DenseGraph, SparseGraph


class Base:
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

        >>> from pecanpy import node2vec
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

    def __init__(self, p, q, workers, verbose, extend=False):
        """Initializ node2vec base class.

        Args:
            p (float): return parameter, value less than 1 encourages returning
                back to previous vertex, and discourage for value grater than 1.
            q (float): in-out parameter, value less than 1 encourages walks to
                go "outward", and value greater than 1 encourage walking within
                a localized neighborhood.
            workers (int):  number of threads to be spawned for runing node2vec
                including walk generation and word2vec embedding.
            verbose (bool): (not implemented yet due to issue with numba jit)
                whether or not to display walk generation progress.
            extend (bool): ``True`` if use node2vec+ extension, default is ``False``

        TODO:
            * Fix numba threads, now uses all possible threads instead of the
                specified number of workers.
            * Think of a way to implement progress monitoring (for ``verbose``)
                during walk generation.

        """
        super(Base, self).__init__()
        self.p = p
        self.q = q
        self.workers = workers
        self.verbose = verbose
        self.extend = extend

    def simulate_walks(self, num_walks, walk_length):
        """Generate walks starting from each nodes ``num_walks`` time.

        Note:
            This is the master process that spawns worker processes, where the
            worker function ``node2vec_walks`` genearte a single random walk
            starting from a vertex of the graph.

        Args:
            num_walks (int): number of walks starting from each node.
            walks_length (int): length of walk.

        """
        num_nodes = len(self.IDlst)
        nodes = np.array(range(num_nodes), dtype=np.uint32)
        start_node_idx_ary = np.concatenate([nodes] * num_walks)
        np.random.shuffle(start_node_idx_ary)

        move_forward = self.get_move_forward()
        has_nbrs = self.get_has_nbrs()
        verbose = self.verbose

        @jit(parallel=True, nogil=True, nopython=True)
        def node2vec_walks():
            """Simulate a random walk starting from start node."""
            n = start_node_idx_ary.size
            # use last entry of each walk index array to keep track of effective walk length
            walk_idx_mat = np.zeros((n, walk_length + 2), dtype=np.uint32)
            walk_idx_mat[:, 0] = start_node_idx_ary  # initialize seeds
            walk_idx_mat[:, -1] = walk_length + 1  # set to full walk length by default

            # progress bar parameters
            n_checkpoints = 10
            checkpoint = n / get_num_threads() // n_checkpoints
            progress_bar_length = 25
            private_count = 0

            for i in prange(n):
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

                if verbose:
                    # TODO: make monitoring less messy
                    private_count += 1
                    if private_count % checkpoint == 0:
                        progress = private_count / n * progress_bar_length * get_num_threads()

                        # manuual construct progress bar since string formatting not supported
                        progress_bar = '|'
                        for k in range(progress_bar_length):
                            progress_bar += '#' if k < progress else ' '
                        progress_bar += '|'

                        print("Thread # " if _get_thread_id() < 10 else "Thread #",
                              _get_thread_id(), "progress:", progress_bar,
                              get_num_threads() * private_count * 10000 // n / 100, "%")

            return walk_idx_mat

        walks = [[self.IDlst[idx] for idx in walk[:walk[-1]]] for walk in node2vec_walks()]

        return walks

    def setup_get_normalized_probs(self):
        """Transition probability computation setup.

        This is function performs necessary preprocessing of computing the
        average edge weights array, which is used later by the transition
        probability computation function ``get_extended_normalized_probs``,
        if node2vec+ is used. Otherwise, return the normal transition function
        ``get_noramlized_probs`` with a trivial placeholder for average edge
        weights array ``avg_wts``.

        """
        if self.extend:  # use n2v+
            get_normalized_probs = self.get_extended_normalized_probs
            avg_wts = self.get_average_weights()
        else:  # use normal n2v
            get_normalized_probs = self.get_normalized_probs
            avg_wts = None
        return get_normalized_probs, avg_wts

    def preprocess_transition_probs(self):
        """Null default preprocess method."""
        pass

    def embed(self, dim=128, num_walks=10, walk_length=80, window_size=10, epochs=1):
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

        Return:
            numpy.ndarray: The embedding matrix, each row is a node embedding
                vector. The index is the same as that for the graph.

        """
        walks = self.simulate_walks(num_walks=num_walks, walk_length=walk_length)
        w2v = Word2Vec(walks, vector_size=dim, window=window_size, sg=1,
                       min_count=0, workers=self.workers, epochs=epochs)

        # index mapping back to node IDs
        idx_list = [w2v.wv.get_index(i) for i in self.IDlst]

        return w2v.wv.vectors[idx_list]


class PreComp(Base, SparseGraph):
    """Precompute transition probabilites.

    This implementation precomputes and store 2nd order transition probabilites
    first and uses read off transition probabilities during the process of
    random walk. The graph type used is ``SparseGraph``.

    Note:
        Need to call ``preprocess_transition_probs()`` first before generating
        walks.

    """

    def __init__(self, p, q, workers, verbose, extend=False):
        """Initialize PreComp mode node2vec."""
        Base.__init__(self, p, q, workers, verbose, extend)

    def get_move_forward(self):
        """Wrap ``move_forward``.

        This function returns a ``numba.jit`` compiled function that takes
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

        @jit(nopython=True, nogil=True)
        def move_forward(cur_idx, prev_idx=None):
            """Move to next node based on transition probabilities."""
            if prev_idx is None:
                normalized_probs = get_normalized_probs(
                    data, indices, indptr, p, q, cur_idx, None, None)
                cdf = np.cumsum(normalized_probs)
                choice = np.searchsorted(cdf, np.random.random())
            else:
                # find index of neighbor for reading alias
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
        """Precompute and store 2nd order transition probabilities."""
        data = self.data
        indices = self.indices
        indptr = self.indptr
        p = self.p
        q = self.q

        get_normalized_probs, avg_wts = self.setup_get_normalized_probs()

        n_nodes = self.indptr.size - 1  # number of nodes
        n = self.indptr[1:] - self.indptr[:-1]  # number of nbrs per node
        n2 = np.power(n, 2)  # number of 2nd order trans probs per node

        self.alias_dim = alias_dim = n
        # use 64 bit unsigned int to prevent overfloating of alias_indptr
        self.alias_indptr = alias_indptr = np.zeros(self.indptr.size, dtype=np.uint64)
        alias_indptr[1:] = np.cumsum(n2)
        n_probs = alias_indptr[-1]  # total number of 2nd order transition probs

        @jit(parallel=True, nopython=True, nogil=True)
        def compute_all_transition_probs():
            alias_j = np.zeros(n_probs, dtype=np.uint32)
            alias_q = np.zeros(n_probs, dtype=np.float64)

            for idx in range(n_nodes):
                offset = alias_indptr[idx]
                dim = alias_dim[idx]

                nbrs = indices[indptr[idx]: indptr[idx + 1]]
                for nbr_idx in prange(n[idx]):
                    nbr = nbrs[nbr_idx]
                    probs = get_normalized_probs(data, indices, indptr, p, q, idx, nbr, avg_wts)

                    start = offset + dim * nbr_idx
                    j_tmp, q_tmp = alias_setup(probs)

                    for i in range(dim):
                        alias_j[start + i] = j_tmp[i]
                        alias_q[start + i] = q_tmp[i]

            return alias_j, alias_q

        self.alias_j, self.alias_q = compute_all_transition_probs()


class SparseOTF(Base, SparseGraph):
    """Sparse graph transition on the fly.

    This implementation do *NOT* precompute transition probabilities in advance
    but instead calculate them on-the-fly during the process of random walk.
    The graph type used is ``SparseGraph``.

    """

    def __init__(self, p, q, workers, verbose, extend=False):
        """Initialize PreComp mode node2vec."""
        Base.__init__(self, p, q, workers, verbose, extend)

    def get_move_forward(self):
        """Wrap ``move_forward``.

        This function returns a ``numba.jit`` compiled function that takes
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

        get_normalized_probs, avg_wts = self.setup_get_normalized_probs()

        @jit(nopython=True, nogil=True)
        def move_forward(cur_idx, prev_idx=None):
            """Move to next node."""
            normalized_probs = get_normalized_probs(
                # data, indices, indptr, p, q, cur_idx, prev_idx)
                data, indices, indptr, p, q, cur_idx, prev_idx, avg_wts)
            cdf = np.cumsum(normalized_probs)
            choice = np.searchsorted(cdf, np.random.random())

            return indices[indptr[cur_idx] + choice]

        return move_forward


class DenseOTF(Base, DenseGraph):
    """Dense graph transition on the fly.

    This implementation do *NOT* precompute transition probabilities in advance
    but instead calculate them on-the-fly during the process of random walk.
    The graph type used is ``DenseGraph``.

    """

    def __init__(self, p, q, workers, verbose, extend=False):
        """Initialize DenseOTF mode node2vec."""
        Base.__init__(self, p, q, workers, verbose, extend)

    def get_move_forward(self):
        """Wrap ``move_forward``.

        This function returns a ``numba.jit`` compiled function that takes
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

        get_normalized_probs, avg_wts = self.setup_get_normalized_probs()

        @jit(nopython=True, nogil=True)
        def move_forward(cur_idx, prev_idx=None):
            """Move to next node."""
            normalized_probs = get_normalized_probs(
                data, nonzero, p, q, cur_idx, prev_idx, avg_wts)
            cdf = np.cumsum(normalized_probs)
            choice = np.searchsorted(cdf, np.random.random())
            nbrs = np.where(nonzero[cur_idx])[0]

            return nbrs[choice]

        return move_forward


@jit(nopython=True, nogil=True)
def alias_setup(probs):
    """Construct alias lookup table.

    This code is modified from the blog post here:
    https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    , where you can find more details about how the method work. In general,
    the alias method improves the time complexity of sampling from a discrete
    random distribution to O(1) if the alias table is setup in advance.

    Args:
        probs (list(float64)): normalized transition probabilities array, could
            be in either list or numpy.ndarray, of float64 values.

    """
    k = probs.size
    q = np.zeros(k, dtype=np.float64)
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


@jit(nopython=True, nogil=True)
def alias_draw(j, q):
    """Draw sample from a non-uniform discrete distribution using alias sampling."""
    k = j.size

    kk = np.random.randint(k)
    if np.random.rand() < q[kk]:
        return kk
    else:
        return j[kk]
