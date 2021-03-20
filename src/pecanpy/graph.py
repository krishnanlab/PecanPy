"""Lite graph objects used by pecanpy."""

import numpy as np
from numba import boolean, jit


class SparseGraph:
    """Sparse Graph object that stores graph as adjacency list.

    Note:
        By default the ``SparseGraph`` object converts the data to Compact
        Sparse Row (csr) format after reading data from an edge list file
        (``.edg``). This format enables more cache optimized computation.

    Examples:
        Read ``.edg`` file and create ``SparseGraph`` object using ``.read_edg``
        method.

        >>> from pecanpy.graph import SparseGraph
        >>>
        >>> g = SparseGraph() # initialize SparseGraph object
        >>> g.read_edg(path_to_edg_file, weighted=True, directed=False) # read graph from edgelist
        >>>
        >>> dense_mat = g.to_dense() # convert to dense adjacency matrix
        >>>

    """

    def __init__(self):
        """Initialize SparseGraph object."""
        self.data = []
        self.indptr = None
        self.indices = None
        self.IDlst = []
        self.IDmap = {}  # id -> index

    def read_edg(self, edg_fp, weighted, directed, csr=True):
        """Read an edgelist file and create sparse graph.

        Note:
            Implicitly discard zero weighted edges; if the same edge is defined
            multiple times with different edge weights, then the last specified
            weight will be used (warning for such behavior will be printed)

        Args:
            edg_fp (str): path to edgelist file, where the file is tab
                seperated and contains 2 or 3 columns depending on whether
                the input graph is weighted, where the the first column
                contains the source nodes and the second column contains the
                destination nodes that interact with the corresponding source
                nodes.
            weighted (bool): whether the graph is weighted. If unweighted,
                only two columns are expected in the edgelist file, and the
                edge weights are implicitely set to 1 for all interactions. If
                weighted, a third column encoding the weight of the interaction
                in numeric value is expected.
            directed (bool): whether the graph is directed, if undirected, the
                edge connecting from destination node to source node is created
                with same edge weight from source node to destination node.
            csr (bool): whether or not to convert to compact sparse row format
                after finished reading the whole edge list for a more compact
                storage and more optimized cache utilization.

        """
        current_node = 0

        with open(edg_fp, "r") as f:

            for line in f:
                if weighted:
                    id1, id2, weight = line.split("\t")
                    weight = float(weight)
                    if weight == 0:
                        continue
                else:
                    terms = line.split("\t")
                    id1, id2 = terms[0], terms[1]
                    weight = float(1)

                id1 = id1.strip()
                id2 = id2.strip()

                # check if ID exist, add to IDmap if not
                for id_ in id1, id2:
                    if id_ not in self.IDmap:
                        self.IDmap[id_] = current_node
                        self.data.append({})
                        current_node += 1

                idx1, idx2 = self.IDmap[id1], self.IDmap[id2]
                # check if edge exists
                if idx2 in self.data[idx1]:
                    if self.data[idx1][idx2] != weight:
                        print(f"Warning: edge from {id1} to {id2} exists, with value of {self.data[idx1][idx2]:.2f}."
                              + f" Now overwrite to {weight:.2f}.")

                # update edge weight
                self.data[idx1][idx2] = weight
                if not directed:
                    self.data[idx2][idx1] = weight

        self.IDlst = sorted(self.IDmap, key=self.IDmap.get)

        if csr:
            self.to_csr()

    def get_has_nbrs(self):
        """Wrap ``has_nbrs``."""
        indptr = self.indptr

        @jit(nopython=True, nogil=True)
        def has_nbrs(idx):
            return indptr[idx] != indptr[idx + 1]

        return has_nbrs

    @staticmethod
    @jit(nopython=True, nogil=True)
    def get_normalized_probs(data, indices, indptr, p, q, cur_idx, prev_idx=None):
        """Calculate transition probabilities.

        Calculate 2nd order transition probabilities by first finidng the
        neighbors of the current state that are not reachable from the previous
        state, and devide the according edge weights by the in-out parameter
        ``q``. Then devide the edge weight from previous state by the return
        parameter ``p``. Finally, the transition probabilities are computed by
        normalizing the biased edge weights.

        Note:
            If ``prev_idx`` present, calculate 2nd order biased transition,
        otherwise calculate 1st order transition.

        """

        def get_nbrs_idx(idx):
            return indices[indptr[idx]: indptr[idx + 1]]

        def get_nbrs_weight(idx):
            return data[indptr[idx]: indptr[idx + 1]].copy()

        nbrs_idx = get_nbrs_idx(cur_idx)
        unnormalized_probs = get_nbrs_weight(cur_idx)

        if prev_idx is not None:  # 2nd order biased walk
            prev_ptr = np.where(nbrs_idx == prev_idx)[0]  # find previous state index
            src_nbrs_idx = get_nbrs_idx(prev_idx)  # neighbors of previous state
            non_com_nbr = isnotin(nbrs_idx, src_nbrs_idx)  # neighbors of current but not previous
            non_com_nbr[prev_ptr] = False  # exclude previous state for in-out bias

            unnormalized_probs[non_com_nbr] /= q  # apply in-out bias
            unnormalized_probs[prev_ptr] /= p  # apply return bias

        normalized_probs = unnormalized_probs / unnormalized_probs.sum()

        return normalized_probs

    def to_csr(self):
        """Construct compressed sparse row matrix."""
        indptr = np.zeros(len(self.IDlst) + 1, dtype=np.uint32)
        for i, row_data in enumerate(self.data):
            indptr[i + 1] = indptr[i] + len(row_data)

        # last element of indptr indicates the total number of nonzero entries
        indices = np.zeros(indptr[-1], dtype=np.uint32)
        data = np.zeros(indptr[-1], dtype=np.float64)

        for i in reversed(range(len(self.data))):
            start = indptr[i]
            end = indptr[i + 1]

            tmp = self.data.pop()
            sorted_keys = sorted(tmp)

            indices[start:end] = np.fromiter(sorted_keys, dtype=np.uint32)
            data[start:end] = np.fromiter(map(tmp.get, sorted_keys), dtype=np.float64)

        self.indptr = indptr
        self.data = data
        self.indices = indices

    def to_dense(self):
        """Construct dense adjacency matrix.

        Note:
            This method does not return DenseGraph object, but instead return
            dense adjacency matrix as ``numpy.ndarray``, the index is the same
            as that of IDlst.

        Return:
            numpy.ndarray: full adjacency matrix indexed by IDmap as 2d numpy
            array.

        """
        n_nodes = len(self.IDlst)
        mat = np.zeros((n_nodes, n_nodes))

        for src_node, src_nbrs in enumerate(self.data):

            for dst_node in src_nbrs:
                mat[src_node, dst_node] = src_nbrs[dst_node]

        return mat


class DenseGraph:
    """Dense Graph object that stores graph as array.

    Examples:
        Read ``.npz`` files and create ``DenseGraph`` object using ``.read_npz``
        method.

        >>> from pecanpy.graph import DenseGraph
        >>> g = DenseGraph() # initialize DenseGraph object
        >>> g.read_npz(paht_to_npz_file, weighted=True, directed=False) # read graph from npz

        Read ``.edg`` files and create ``DenseGraph`` object using ``.read_edg``
        method.

        >>> from pecanpy.graph import DenseGraph
        >>> g = DenseGraph() # initialize DenseGraph object
        >>> g.read_edg(path_to_edg_file, weighted=True, directed=False) # read graph from edgelist
        >>>
        >>> g.save(npz_outpath) # save the network as npz file, which could be loaded faster if network is dense
        >>>

    """

    def __init__(self):
        """Initialize DenseGraph object."""
        self.data = None
        self.nonzero = None
        self.IDlst = []
        self.IDmap = {}  # id -> index

    def read_npz(self, npz_fp, weighted, directed):
        """Read ``.npz`` file and create dense graph.

        Args:
            npz_fp (str): path to ``.npz`` file.
            weighted (bool): whether the graph is weighted, if unweighted,
                all none zero weights will be converted to 1.
            directed (bool): not used, for compatibility with ``SparseGraph``.

        """
        raw = np.load(npz_fp)
        self.data = raw["data"]
        self.nonzero = self.data != 0
        self.IDlst = list(raw["IDs"])
        self.IDmap = {j: i for i, j in enumerate(self.IDlst)}

    def read_edg(self, edg_fp, weighted, directed):
        """Read an edgelist file and construct dense graph."""
        sparse_graph = SparseGraph()
        sparse_graph.read_edg(edg_fp, weighted, directed, csr=False)

        self.IDlst = sparse_graph.IDlst
        self.IDmap = sparse_graph.IDmap
        self.data = sparse_graph.to_dense()
        self.nonzero = self.data != 0

    def save(self, fp):
        """Save as ``.npz`` file."""
        np.savez(fp, data=self.data, IDs=self.IDlst)

    def get_has_nbrs(self):
        """Wrap ``has_nbrs``."""
        nonzero = self.nonzero

        @jit(nopython=True, nogil=True)
        def has_nbrs(idx):
            for j in range(nonzero.shape[1]):
                if nonzero[idx, j]:
                    return True
            return False

        return has_nbrs

    @staticmethod
    @jit(nopython=True, nogil=True)
    def get_normalized_probs(data, nonzero, p, q, cur_idx, prev_idx=None):
        """Calculate transition probabilities.

        Calculate 2nd order transition probabilities by first finidng the
        neighbors of the current state that are not reachable from the previous
        state, and devide the according edge weights by the in-out parameter
        ``q``. Then devide the edge weight from previous state by the return
        parameter ``p``. Finally, the transition probabilities are computed by
        normalizing the biased edge weights.

        Note:
            If ``prev_idx`` present, calculate 2nd order biased transition,
        otherwise calculate 1st order transition.

        """
        nbrs_ind = nonzero[cur_idx]
        unnormalized_probs = data[cur_idx].copy()

        if prev_idx is not None:  # 2nd order biased walks
            non_com_nbr = np.logical_and(nbrs_ind, ~nonzero[prev_idx])  # nbrs of cur but not prev
            non_com_nbr[prev_idx] = False  # exclude previous state for in-out biase

            unnormalized_probs[non_com_nbr] /= q  # apply in-out bias
            unnormalized_probs[prev_idx] /= p  # apply return bias

        unnormalized_probs = unnormalized_probs[nbrs_ind]
        normalized_probs = unnormalized_probs / unnormalized_probs.sum()

        return normalized_probs


@jit(nopython=True, nogil=True)
def isnotin(ary1, ary2):
    """Value in ary1 but not in ary2.

    Used to find neighbor indices that are in current state but not in the
    previous state, which will be biased using the in-out parameter ``q``. The
    values in each of the two arrays are sorted ascendingly. The main idea is
    to scan through ``ary1`` and compare the values in ``ary2`` in a way that
    at most one pass of each array is needed instead of a nested loop (for
    each element in ``ary1``, compare against every element in ``ary2``),
    which is much more efficient. Checkout the following example for more
    intuition.

    Examples:
        Consider the following example with two arrays, the ``*`` above
        ``ary1`` and ``ary2`` indicate the pointers (``ptr1`` and ``ptr2``
        respectively).

        >>> ary1 = [1, 2, 5]
        >>> ary2 = [1, 5]
        >>>
        >>> # iteration1: indicator = [False, True, True]
        >>>  *
        >>> [1, 2, 5]
        >>>  *
        >>> [1, 5]
        >>>
        >>> # iteration2: indicator = [False, True, True]
        >>>     *
        >>> [1, 2, 5]
        >>>     *
        >>> [1, 5]
        >>>
        >>> # iteration3: indicator = [False, True, False]
        >>>        *
        >>> [1, 2, 5]
        >>>     *
        >>> [1, 5]
        >>>
        >>> # end of loop

        Note that this does not remove the index of the previous state.
        Instead, the index of the previous state will be removed once the
        indicator is returned to the ``get_normalized_probs``.


    """
    indicator = np.ones(ary1.size, dtype=boolean)
    ptr2 = 0
    for ptr1 in range(ary1.size):
        if ptr2 == ary2.size:  # end of ary2
            break

        val1 = ary1[ptr1]
        val2 = ary2[ptr2]

        if val1 < val2:
            continue

        elif val1 == val2:  # found a matching value
            indicator[ptr1] = False
            ptr2 += 1

        elif val1 > val2:
            for j in range(ptr2, ary2.size):
                if ary2[j] == val1:
                    indicator[ptr1] = False
                    ptr2 += 1
                    break

                elif ary2[j] > val1:
                    ptr2 = j
                    break

            continue

    return indicator
