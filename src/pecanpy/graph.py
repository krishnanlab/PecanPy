"""Lite graph objects used by pecanpy."""

import numpy as np


class IDHandle:
    """Node ID handler."""

    def __init__(self):
        """Initialize ID list and ID map."""
        self.IDlst = []
        self.IDmap = {}  # id -> index

    def set_ids(self, ids):
        """Update ID list and mapping.

        Set IDlst given the input ids and also set the IDmap based on it.
        """
        self.IDlst = ids
        self.IDmap = {j: i for i, j in enumerate(ids)}


class SparseGraph(IDHandle):
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
        >>> # initialize SparseGraph object
        >>> g = SparseGraph()
        >>>
        >>> # read graph from edgelist
        >>> g.read_edg(path_to_edg_file, weighted=True, directed=False)
        >>>
        >>> dense_mat = g.to_dense() # convert to dense adjacency matrix
        >>>
        >>> # save the csr graph as npz file to be used later
        >>> g.save(npz_outpath)

    """

    def __init__(self):
        """Initialize SparseGraph object."""
        super(SparseGraph, self).__init__()
        self.data = []
        self.indptr = None
        self.indices = None

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
                        print(
                            f"WARNING: edge from {id1} to {id2} exists, with "
                            f"value of {self.data[idx1][idx2]:.2f}. "
                            f"Now overwrite to {weight:.2f}.",
                        )

                # update edge weight
                self.data[idx1][idx2] = weight
                if not directed:
                    self.data[idx2][idx1] = weight

        self.IDlst = sorted(self.IDmap, key=self.IDmap.get)

        if csr:
            self.to_csr()

    def read_npz(self, fp, weighted, directed):
        """Directly read a CSR sparse graph.

        Note:
            To generate a CSR file compatible with PecanPy, first load the graph
                as a sparse graph using the SparseGraph (with ``csr=True``).
                Then save the sparse graph to a csr file using the ``save``
                method from ``SparseGraph``. The saved ``.npz`` file can then
                be loaded directly by ``SparseGraph`` later.

        Args:
            fp (str): path to the csr file, which is an npz file with four
                arrays with keys 'IDs', 'data', 'indptr', 'indices', which
                correspond to the node IDs, the edge weights, the offset array
                for each node, and the indices of the edges.
            weighted (bool): whether the graph is weighted, if unweighted,
                all edge weights will be converted to 1.
            directed (bool): not used, for compatibility with ``SparseGraph``.

        """
        raw = np.load(fp)
        self.set_ids(raw["IDs"].tolist())
        self.data = raw["data"]
        if not weighted:  # overwrite edge weights with constant
            self.data[:] = 1.0
        self.indptr = raw["indptr"]
        self.indices = raw["indices"]

    def save(self, fp):
        """Save CSR as ``.csr.npz`` file."""
        np.savez(
            fp,
            IDs=self.IDlst,
            data=self.data,
            indptr=self.indptr,
            indices=self.indices,
        )

    def from_mat(self, adj_mat, ids):
        """Construct graph using adjacency matrix and node ids.

        Args:
            adj_mat(:obj:`numpy.ndarray`): 2D numpy array of adjacency matrix
            ids(:obj:`list` of str): node ID list

        """
        data = []  # construct edge list
        for row in adj_mat:
            data.append({})
            for j, weight in enumerate(row):
                if weight != 0:
                    data[-1][j] = weight

        # save edgelist and id data and convert to csr format
        self.data = data
        self.set_ids(ids)
        self.to_csr()

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


class DenseGraph(IDHandle):
    """Dense Graph object that stores graph as array.

    Examples:
        Read ``.npz`` files and create ``DenseGraph`` object using ``read_npz``.

        >>> from pecanpy.graph import DenseGraph
        >>> g = DenseGraph() # initialize DenseGraph object
        >>> g.read_npz(paht_to_npz_file, weighted=True, directed=False)

        Read ``.edg`` files and create ``DenseGraph`` object using ``read_edg``.

        >>> from pecanpy.graph import DenseGraph
        >>>
        >>> # initialize DenseGraph object
        >>> g = DenseGraph()
        >>>
        >>> # read graph from edgelist
        >>> g.read_edg(path_to_edg_file, weighted=True, directed=False)
        >>>
        >>> # save the dense graph as npz file to be used later
        >>> g.save(npz_outpath)

    """

    def __init__(self):
        """Initialize DenseGraph object."""
        super(DenseGraph, self).__init__()
        self.data = None
        self.nonzero = None

    def read_npz(self, fp, weighted, directed):
        """Read ``.npz`` file and create dense graph.

        Args:
            fp (str): path to ``.npz`` file.
            weighted (bool): whether the graph is weighted, if unweighted,
                all none zero weights will be converted to 1.
            directed (bool): not used, for compatibility with ``SparseGraph``.

        """
        raw = np.load(fp)
        self.data = raw["data"]
        self.nonzero = self.data != 0
        if not weighted:  # overwrite edge weights with constant
            self.data = self.nonzero * 1.0
        self.set_ids(raw["IDs"].tolist())

    def read_edg(self, edg_fp, weighted, directed):
        """Read an edgelist file and construct dense graph."""
        sparse_graph = SparseGraph()
        sparse_graph.read_edg(edg_fp, weighted, directed, csr=False)

        self.set_ids(sparse_graph.IDlst)
        self.data = sparse_graph.to_dense()
        self.nonzero = self.data != 0

    def from_mat(self, adj_mat, ids):
        """Construct graph using adjacency matrix and node ids.

        Args:
            adj_mat(:obj:`numpy.ndarray`): 2D numpy array of adjacency matrix
            ids(:obj:`list` of str): node ID list

        """
        self.data = adj_mat
        self.nonzero = adj_mat != 0
        self.set_ids(ids)

    def save(self, fp):
        """Save dense graph  as ``.dense.npz`` file."""
        np.savez(fp, data=self.data, IDs=self.IDlst)
