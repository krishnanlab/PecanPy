"""Lite graph objects used by pecanpy."""
from typing import Iterator
from typing import List
from typing import Tuple

import numpy as np


class BaseGraph:
    """Base Graph object.

    Handles node id and provides general properties including num_nodes,
    and density. The num_edges property is to be specified by the derived
    graph objects.

    """

    def __init__(self):
        """Initialize ID list and ID map."""
        self.IDlst = []
        self.IDmap = {}  # id -> index

    @property
    def nodes(self):
        """Return the list of node IDs."""
        return self.IDlst

    @property
    def num_nodes(self):
        """Return the number of nodes in the graph."""
        return len(self.IDlst)

    @property
    def num_edges(self):
        """Return the number of edges in the graph."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not have num_edges, use the "
            f"derived classes like SparseGraph and DenseGraph instead.",
        )

    @property
    def density(self):
        """Return the edge density of the graph."""
        return self.num_edges / self.num_nodes / (self.num_nodes - 1)

    def set_ids(self, ids):
        """Update ID list and mapping.

        Set IDlst given the input ids and also set the IDmap based on it.

        """
        self.IDlst = ids
        self.IDmap = {j: i for i, j in enumerate(ids)}


class AdjlstGraph(BaseGraph):
    """Adjacency list Graph object used for reading/writing edge list files.

    Sparse Graph object that stores graph as adjacency list.

    Note:
        AdjlstGraph is only used for reading/writing edge list files and do not
        support random walk computations since Numba njit do not work with
        Python data structures like list and dict.

    Examples:
        Read ``.edg`` file and create ``SparseGraph`` object using ``.read_edg``
        method.

        >>> from pecanpy.graph import AdjlstGraph
        >>>
        >>> # initialize SparseGraph object
        >>> g = AdjlstGraph()
        >>>
        >>> # read graph from edgelist
        >>> g.read(path_to_edg_file, weighted=True, directed=False)
        >>>
        >>> indptr, indices, data = g.to_csr()  # convert to csr
        >>>
        >>> dense_mat = g.to_dense()  # convert to dense adjacency matrix
        >>>
        >>> g.save(edg_outpath)  # save the graph to an edge list file

    """

    def __init__(self):
        """Initialize AdjlstGraph object."""
        super().__init__()
        self._data = []  # list of dict of node_indexx -> edge_weight
        self._num_edges = 0

    @property
    def edges_iter(self) -> Iterator[Tuple[int, int, float]]:
        """Return an iterator that iterates over all edges."""
        for head, head_nbrs in enumerate(self._data):
            for tail in sorted(head_nbrs):
                yield head, tail, head_nbrs[tail]

    @property
    def edges(self) -> List[Tuple[int, int, float]]:
        """Return a list of triples (head, tail, weight) representing edges."""
        return list(self.edges_iter)

    @property
    def num_edges(self):
        """Return the number of edges in the graph."""
        return self._num_edges

    @staticmethod
    def _read_edge_line(edge_line, weighted, delimiter):
        """Read a line from the edge list file."""
        terms = edge_line.strip().split(delimiter)
        id1, id2 = terms[0].strip(), terms[1].strip()

        weight = 1.0
        if weighted:
            if len(terms) != 3:
                raise ValueError(
                    f"Expecting three columns in the edge list file for a "
                    f"weighted graph, got {len(terms)} instead: {edge_line!r}",
                )
            weight = float(terms[-1])

        return id1, id2, weight

    @staticmethod
    def _is_valid_edge_weight(id1, id2, weight):
        """Check if the edge weight is non-negative."""
        if weight <= 0:
            edg_str = f"w({id1},{id2}) = {weight}"
            print(f"WARNING: non-positive edge ignored: {edg_str}")
            return False
        return True

    def _check_edge_existence(self, id1, id2, idx1, idx2, weight):
        """Check if an edge exists.

        If the edge to be added already exists and the new edge weight is
        different from the existing edge weights, print warning message.

        """
        if idx2 in self._data[idx1]:
            if self._data[idx1][idx2] != weight:
                print(
                    f"WARNING: edge from {id1} to {id2} exists, with "
                    f"value of {self._data[idx1][idx2]:.2f}. "
                    f"Now overwrite to {weight:.2f}.",
                )

    def get_node_idx(self, node_id):
        """Get index of the node and create new node when necessary."""
        self.add_node(node_id)
        return self.IDmap[node_id]

    def add_node(self, node_id):
        """Create a new node.

        Add a new node to the graph if not already exsitsed, by updating the
        ID list, ID map, and the adjacency list data. Otherwise pass through
        without further actions.

        Note:
            Does not raise error even if the node alrealy exists.

        """
        if node_id not in self.IDmap:
            self.IDmap[node_id] = self.num_nodes
            self.IDlst.append(node_id)
            self._data.append({})

    def _add_edge_from_idx(self, idx1: int, idx2: int, weight: float):
        """Add an edge based on the head and tail node index with weight."""
        self._data[idx1][idx2] = weight
        self._num_edges += 1

    def add_edge(self, id1, id2, weight=1.0, directed=False):
        """Add an edge to the graph.

        Note:
            Non-positive edges are ignored.

        Args:
            id1 (str): first node id.
            id2 (str): second node id.
            weight (float): the edge weight, default is 1.0
            directed (bool): whether the edge is directed or not.

        """
        if self._is_valid_edge_weight(id1, id2, weight):
            idx1, idx2 = map(self.get_node_idx, (id1, id2))
            self._check_edge_existence(id1, id2, idx1, idx2, weight)

            self._add_edge_from_idx(idx1, idx2, weight)
            if not directed:
                self._add_edge_from_idx(idx2, idx1, weight)

    def read(self, edg_fp, weighted, directed, delimiter="\t"):
        """Read an edgelist file and create sparse graph.

        Note:
            Implicitly discard zero weighted edges; if the same edge is defined
            multiple times with different edge weights, then the last specified
            weight will be used (warning for such behavior will be printed).

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
            delimiter (str): delimiter of the edge list file, default is tab.

        """
        with open(edg_fp, "r") as f:
            for edge_line in f:
                edge = self._read_edge_line(edge_line, weighted, delimiter)
                self.add_edge(*edge, directed)

    def save(self, fp: str, unweighted: bool = False, delimiter: str = "\t"):
        """Save AdjLst as an ``.edg`` edge list file.

        Args:
            unweighted (bool): If set to True, only write two columns,
                corresponding to the head and tail nodes of the edges, and
                ignore the edge weights (default: :obj:`False`).
            delimiter (str): Delimiter for separating fields.

        """
        with open(fp, "w") as f:
            for h, t, w in self.edges_iter:
                h_id, t_id = self.nodes[h], self.nodes[t]
                terms = (h_id, t_id) if unweighted else (h_id, t_id, str(w))
                f.write(f"{delimiter.join(terms)}\n")

    def to_csr(self):
        """Construct compressed sparse row matrix."""
        indptr = np.zeros(len(self.IDlst) + 1, dtype=np.uint32)
        for i, row_data in enumerate(self._data):
            indptr[i + 1] = indptr[i] + len(row_data)

        # last element of indptr indicates the total number of nonzero entries
        indices = np.zeros(indptr[-1], dtype=np.uint32)
        data = np.zeros(indptr[-1], dtype=np.float32)

        for i, nbrs in enumerate(self._data):
            if len(nbrs) == 0:
                continue
            new_indices, new_data = zip(*[(j, nbrs[j]) for j in sorted(nbrs)])
            chunk = slice(indptr[i], indptr[i + 1])
            indices[chunk] = np.array(new_indices, dtype=np.uint32)
            data[chunk] = np.array(new_data, dtype=np.float32)

        return indptr, indices, data

    def to_dense(self):
        """Construct dense adjacency matrix.

        Note:
            This method does not return DenseGraph object, but instead return
            dense adjacency matrix as ``numpy.ndarray``, the index is the same
            as that of IDlst.

        Return:
            numpy.ndarray: Full adjacency matrix as 2d numpy array.

        """
        n_nodes = len(self.IDlst)
        mat = np.zeros((n_nodes, n_nodes))

        for src_node, src_nbrs in enumerate(self._data):
            for dst_node in src_nbrs:
                mat[src_node, dst_node] = src_nbrs[dst_node]

        return mat

    @classmethod
    def from_mat(cls, adj_mat, node_ids, **kwargs):
        """Construct graph using adjacency matrix and node ids.

        Args:
            adj_mat(:obj:`numpy.ndarray`): 2D numpy array of adjacency matrix
            node_ids(:obj:`list` of str): node ID list

        Return:
            An adjacency graph object representing the adjacency matrix.

        """
        g = cls(**kwargs)

        # Setup node idmap in the order of node_ids
        for node_id in node_ids:
            g.add_node(node_id)

        # Fill in edge data
        for idx1, idx2 in zip(*np.where(adj_mat != 0)):
            g._add_edge_from_idx(idx1, idx2, adj_mat[idx1, idx2])

        return g


class SparseGraph(BaseGraph):
    """Sparse Graph object that stores graph as adjacency list.

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
        >>> # save the csr graph as npz file to be used later
        >>> g.save(npz_outpath)

    """

    def __init__(self):
        """Initialize SparseGraph object."""
        super().__init__()
        self.data = self.indptr = self.indices = None

    @property
    def num_edges(self):
        """Return the number of edges in the graph."""
        return self.indptr[-1]

    def read_edg(self, edg_fp, weighted, directed, delimiter="\t"):
        """Create CSR sparse graph from edge list.

        First create ``AdjlstGraph`` by reading the edge list file, and then
        convert to ``SparseGraph`` via ``to_csr``.

        Args:
            edg_fp (str): path to edgelist file.
            weighted (bool): whether the graph is weighted.
            directed (bool): whether the graph is directed.
            delimiter (str): delimiter used between node IDs.

        """
        g = AdjlstGraph()
        g.read(edg_fp, weighted, directed, delimiter)
        self.set_ids(g.IDlst)
        self.indptr, self.indices, self.data = g.to_csr()

    def read_npz(self, fp, weighted):
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

    @classmethod
    def from_adjlst_graph(cls, adjlst_graph, **kwargs):
        """Construct csr graph from adjacency list graph.

        Args:
            adjlst_graph (:obj:`pecanpy.graph.AdjlstGraph`): Adjacency list
                graph to be converted.

        """
        g = cls(**kwargs)
        g.set_ids(adjlst_graph.IDlst)
        g.indptr, g.indices, g.data = adjlst_graph.to_csr()
        return g

    @classmethod
    def from_mat(cls, adj_mat, node_ids, **kwargs):
        """Construct csr graph using adjacency matrix and node ids.

        Note:
            Only consider positive valued edges.

        Args:
            adj_mat(:obj:`numpy.ndarray`): 2D numpy array of adjacency matrix
            node_ids(:obj:`list` of str): node ID list

        """
        g = cls(**kwargs)
        g.set_ids(node_ids)
        adjlst_graph = AdjlstGraph.from_mat(adj_mat, node_ids)
        g.indptr, g.indices, g.data = adjlst_graph.to_csr()
        return g


class DenseGraph(BaseGraph):
    """Dense Graph object that stores graph as array.

    Examples:
        Read ``.npz`` files and create ``DenseGraph`` object using ``read_npz``.

        >>> from pecanpy.graph import DenseGraph
        >>>
        >>> g = DenseGraph() # initialize DenseGraph object
        >>>
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
        super().__init__()
        self._data = None
        self._nonzero = None

    @property
    def num_edges(self):
        """Return the number of edges in the graph."""
        return self.nonzero.sum()

    @property
    def data(self):
        """Return the adjacency matrix."""
        return self._data

    @property
    def nonzero(self):
        """Return the nonzero mask for the adjacency matrix."""
        return self._nonzero

    @data.setter
    def data(self, data):
        """Set adjacency matrix and the corresponding nonzero matrix."""
        self._data = data.astype(float)
        self._nonzero = self._data != 0

    def read_npz(self, fp, weighted):
        """Read ``.npz`` file and create dense graph.

        Args:
            fp (str): path to ``.npz`` file.
            weighted (bool): whether the graph is weighted, if unweighted,
                all none zero weights will be converted to 1.

        """
        raw = np.load(fp)
        self.data = raw["data"]
        if not weighted:  # overwrite edge weights with constant
            self.data = self.nonzero * 1.0
        self.set_ids(raw["IDs"].tolist())

    def read_edg(self, edg_fp, weighted, directed, delimiter="\t"):
        """Read an edgelist file and construct dense graph."""
        g = AdjlstGraph()
        g.read(edg_fp, weighted, directed, delimiter)

        self.set_ids(g.IDlst)
        self.data = g.to_dense()

    def save(self, fp):
        """Save dense graph  as ``.dense.npz`` file."""
        np.savez(fp, data=self.data, IDs=self.IDlst)

    @classmethod
    def from_adjlst_graph(cls, adjlst_graph, **kwargs):
        """Construct dense graph from adjacency list graph.

        Args:
            adjlst_graph (:obj:`pecanpy.graph.AdjlstGraph`): Adjacency list
                graph to be converted.

        """
        g = cls(**kwargs)
        g.set_ids(adjlst_graph.IDlst)
        g.data = adjlst_graph.to_dense()
        return g

    @classmethod
    def from_mat(cls, adj_mat, node_ids, **kwargs):
        """Construct dense graph using adjacency matrix and node ids.

        Args:
            adj_mat(:obj:`numpy.ndarray`): 2D numpy array of adjacency matrix
            ids(:obj:`list` of str): node ID list

        """
        g = cls(**kwargs)
        g.data = adj_mat
        g.set_ids(node_ids)
        return g
