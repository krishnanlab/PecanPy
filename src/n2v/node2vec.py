import numpy as np
from numba import jit, prange, boolean
from n2v.graph import SparseGraph, DenseGraph

class Base:
    """Improved version of original node2vec
    Parallelized transition probabilities pre-computation and random walks

    """
    def __init__(self, p, q, workers, verbose):
        super(Base, self).__init__()
        self.p = p
        self.q = q
        self.workers = workers
        self.verbose = verbose


    def simulate_walks(self, num_walks, walk_length):
        """Generate walks starting from each nodes `num_walks` time

        Notes:
            This is master worker

        Args:
            num_walks(int): number of walks starting from each node
            walks_length(int): length of walk

        """
        num_nodes = len(self.IDlst)
        nodes = np.array(range(num_nodes), dtype=np.uint32)
        start_node_idx_ary = np.concatenate([nodes]*num_walks)
        np.random.shuffle(start_node_idx_ary)

        p = self.p
        q = self.q
        move_forward = self.get_move_forward()
        has_nbrs = self.get_has_nbrs()

        @jit(parallel=True, nogil=True, nopython=True)
        def node2vec_walks():
            """Simulate a random walk starting from start node."""
            n = start_node_idx_ary.size
            walk_idx_mat = np.zeros((n, walk_length + 1), dtype=np.uint32)
            walk_idx_mat[:,0] = start_node_idx_ary
            for i in prange(n):
                start_node_idx = walk_idx_mat[i,0]
                walk_idx_mat[i,1] = move_forward(start_node_idx)
                #TODO: print status in regular interval

                for j in range(2, walk_length + 1):
                    cur_idx = walk_idx_mat[i, j - 1]
                    if has_nbrs(cur_idx):
                        prev_idx = walk_idx_mat[i, j - 2]
                        walk_idx_mat[i,j] = move_forward(cur_idx, prev_idx)
                    else:
                        print("Dead end!")
                        break

            return walk_idx_mat


        walks = [[self.IDlst[idx] for idx in walk] for walk in node2vec_walks()]

        return walks


    def preprocess_transition_probs(self):
        pass


class PreComp(Base, SparseGraph):
    def __init__(self, p, q, workers, verbose):
        Base.__init__(self, p, q, workers, verbose)


    def get_move_forward(self):
        """Wrapper for `move_forward` that calculates transition probabilities
        and draw next node from the distribution"""
        data = self.data
        indices = self.indices
        indptr = self.indptr
        p = self.p
        q = self.q
        get_normalized_probs = self.get_normalized_probs

        alias_J = self.alias_J
        alias_q = self.alias_q
        alias_indptr = self.alias_indptr
        alias_dim = self.alias_dim

        @jit(nopython=True, nogil=True)
        def move_forward(cur_idx, prev_idx=None):
            if prev_idx is None:
                normalized_probs = get_normalized_probs(data, indices, indptr, 
                                                        p, q, cur_idx)
                cdf = np.cumsum(normalized_probs)
                choice = np.searchsorted(cdf, np.random.random())
            else:
                # find index of neighbor for reading alias
                start = indptr[cur_idx]
                end = indptr[cur_idx+1]
                nbr_idx = np.searchsorted(indices[start:end], prev_idx)
                if indices[start+nbr_idx] != prev_idx:
                    print("FATAL ERROR! Neighbor not found.")

                dim = alias_dim[cur_idx]
                start = alias_indptr[cur_idx] + dim * nbr_idx
                end = start + dim
                choice = alias_draw(alias_J[start:end], alias_q[start:end])

            return indices[indptr[cur_idx] + choice]

        return move_forward


    def preprocess_transition_probs(self):
        """Precompute and store 2nd order transition probabilities as array of
        dense matrices

        """

        data = self.data
        indices = self.indices
        indptr = self.indptr
        p = self.p
        q = self.q
        get_normalized_probs = self.get_normalized_probs

        N = self.indptr.size - 1 # number of nodes
        n = self.indptr[1:] - self.indptr[:-1] # number of nbrs per node
        n2 = np.power(n, 2) # number of 2nd order trans probs per node

        self.alias_dim = alias_dim = n
        # use 64 bit unsigned int to prevent overfloating of alias_indptr
        self.alias_indptr = alias_indptr = np.zeros(self.indptr.size, dtype=np.uint64)
        alias_indptr[1:] = np.cumsum(n2)
        n_probs = alias_indptr[-1] # total number of 2nd order transition probs

        @jit(parallel=True, nopython=True, nogil=True)
        def compute_all_transition_probs():
            alias_J = np.zeros(n_probs, dtype=np.uint32)
            alias_q = np.zeros(n_probs, dtype=np.float64)

            for idx in range(N):
                offset = alias_indptr[idx]
                dim = alias_dim[idx]

                nbrs = indices[indptr[idx]:indptr[idx+1]]
                for nbr_idx in prange(n[idx]):
                    nbr = nbrs[nbr_idx]
                    probs = get_normalized_probs(data, indices, indptr, p, q, idx, nbr)

                    start = offset + dim * nbr_idx
                    J_tmp, q_tmp = alias_setup(probs)

                    for i in range(dim):
                        alias_J[start+i] = J_tmp[i]
                        alias_q[start+i] = q_tmp[i]

            return alias_J, alias_q

        self.alias_J, self.alias_q = compute_all_transition_probs()


class SparseOTF(Base, SparseGraph):
    def __init__(self, p, q, workers, verbose):
        Base.__init__(self, p, q, workers, verbose)


    def get_move_forward(self):
        """Wrapper for `move_forward` that calculates transition probabilities
        and draw next node from the distribution"""
        data = self.data
        indices = self.indices
        indptr = self.indptr
        p = self.p
        q = self.q
        get_normalized_probs = self.get_normalized_probs

        @jit(nopython=True, nogil=True)
        def move_forward(cur_idx, prev_idx=None):
            normalized_probs = get_normalized_probs(data, indices, indptr, 
                                                    p, q, cur_idx, prev_idx)
            cdf = np.cumsum(normalized_probs)
            choice = np.searchsorted(cdf, np.random.random())

            return indices[indptr[cur_idx] + choice]

        return move_forward


class DenseOTF(Base, DenseGraph):
    def __init__(self, p, q, workers, verbose):
        Base.__init__(self, p, q, workers, verbose)


    def get_move_forward(self):
        data = self.data
        nonzero = self.nonzero
        p = self.p
        q = self.q
        get_normalized_probs = self.get_normalized_probs

        @jit(nopython=True, nogil=True)
        def move_forward(cur_idx, prev_idx=None):
            normalized_probs = get_normalized_probs(data, nonzero, 
                                                    p, q, cur_idx, prev_idx)
            cdf = np.cumsum(normalized_probs)
            choice = np.searchsorted(cdf, np.random.random())
            nbrs = np.where(nonzero[cur_idx])[0]

            return nbrs[choice]

        return move_forward


@jit(nopython=True, nogil=True)
def alias_setup(probs):
    """Setup lookup table for alias method
    https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/

    Args:
        probs(*float64): normalized transition probabilities array

    """
    K = probs.size
    q = np.zeros(K, dtype=np.float64)
    J = np.zeros(K, dtype=np.uint32)

    smaller = np.zeros(K, dtype=np.uint32)
    larger = np.zeros(K, dtype=np.uint32)
    smaller_ptr = 0
    larger_ptr = 0

    for kk in range(K):
        q[kk] = K * probs[kk]
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

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller[smaller_ptr] = large
            smaller_ptr += 1
        else:
            larger[larger_ptr] = large
            larger_ptr += 1

    return J, q


@jit(nopython=True, nogil=True)
def alias_draw(J, q):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    K = J.size

    kk = np.random.randint(K)
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
