"""Lite graph objects used by pecanpy."""
import warnings

import numpy as np

from .typing import AdjMat
from .typing import AdjNonZeroMat
from .typing import CSR
from .typing import Dict
from .typing import Float32Array
from .typing import Iterator
from .typing import List
from .typing import Optional
from .typing import Sequence
from .typing import Tuple
from .typing import Uint32Array


class BaseGraph:
    """Base Graph object.

    Handles node id and provides general properties including num_nodes,
    and density. The num_edges property is to be specified by the derived
    graph objects.

    """

    def __init__(self):
        """Initialize ID list and ID map."""
        self._node_ids: List[str] = []
        self._node_idmap: Dict[str, int] = {}  # id -> index

    @property
    def nodes(self) -> List[str]:
        """Return the list of node IDs."""
        return self._node_ids

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        """Return the number of edges in the graph."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not have num_edges, use the "
            f"derived classes like SparseGraph and DenseGraph instead.",
        )

    @property
    def density(self) -> float:
        """Return the edge density of the graph."""
        return self.num_edges / self.num_nodes / (self.num_nodes - 1)

    def set_node_ids(
        self,
        node_ids: Optional[Sequence[str]],
        implicit_ids: bool = False,
        num_nodes: Optional[int] = None,
    ):
        """Update ID list and mapping.

        Set _node_ids given the input node IDs and also set the corresponding
        _node_idmap based on it, which maps from node ID to the index.

        Args:
            node_ids (:obj:`list` of :obj:`str`, optional): List of node IDs to
                use. If not available, will implicitly set node IDs to the
                canonical ordering of nodes with a warning message, which is
                suppressed if `implicit_ids` is set to True.
            implicit_ids (bool): Implicitly set the node IDs to the canonical
                node ordering. If set to False and node IDs are not available,
                it will also set implicit node IDs, but with a warning message.
                The warning message can be suppressed if `implicit_ids` is set
                to True as a confirmation of the behavior.
            num_nodes (int, optional): Number of nodes, used when try to set
                implicit node IDs.

        """
        if (node_ids is not None) and (not implicit_ids):
            self._node_ids = list(node_ids)
        elif num_nodes is None:
            raise ValueError(
                "Need to specify `num_nodes` when setting implicit node IDs.",
            )
        else:
            self.set_node_ids(list(map(str, range(num_nodes))))
            if not implicit_ids:
                warnings.warn(
                    "WARNING: Implicitly set node IDs to the canonical node "
                    "ordering due to missing IDs field in the raw CSR npz "
                    "file. This warning message can be suppressed by setting "
                    "implicit_ids to True in the read_npz function call, or "
                    "by setting the --implicit_ids flag in the CLI",
                )
        self._node_idmap = {j: i for i, j in enumerate(self._node_ids)}

    def get_has_nbrs(self):
        """Abstract method to be specified by derived classes."""
        raise NotImplementedError

    def get_move_forward(self):
        """Abstract method to be specified by derived classes."""
        raise NotImplementedError


class AdjlstGraph(BaseGraph):
    """Adjacency list Graph object used for reading/writing edge list files.

    Sparse Graph object that stores graph as adjacency list.

    Note:
        AdjlstGraph is only used for reading/writing edge list files and do not
        support random walk computations since Numba njit do not work with
        Python data structures like list and dict.

    Examples:
        Read ``.edg`` file and create ``SparseGraph`` object using
        ``.read_edg`` method.

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
        self._data: List[Dict[int, float]] = []  # list of nbrs idx -> weights
        self._num_edges: int = 0

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
    def _read_edge_line(
        edge_line: str,
        weighted: bool,
        delimiter: str,
    ) -> Tuple[str, str, float]:
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
    def _is_valid_edge_weight(id1: str, id2: str, weight: float) -> bool:
        """Check if the edge weight is non-negative."""
        if weight <= 0:
            edg_str = f"w({id1},{id2}) = {weight}"
            warnings.warn(
                f"Non-positive edge ignored: {edg_str}",
                RuntimeWarning,
            )
            return False
        return True

    def _check_edge_existence(
        self,
        id1: str,
        id2: str,
        idx1: int,
        idx2: int,
        weight: float,
    ):
        """Check if an edge exists.

        If the edge to be added already exists and the new edge weight is
        different from the existing edge weights, print warning message.

        """
        if idx2 in self._data[idx1] and self._data[idx1][idx2] != weight:
            warnings.warn(
                f"edge from {id1} to {id2} exists, with "
                f"value of {self._data[idx1][idx2]:.2f}. "
                f"Now overwrite to {weight:.2f}.",
                RuntimeWarning,
            )

    def get_node_idx(self, node_id: str) -> int:
        """Get index of the node and create new node when necessary."""
        self.add_node(node_id)
        return self._node_idmap[node_id]

    def add_node(self, node_id: str):
        """Create a new node.

        Add a new node to the graph if not already exsitsed, by updating the
        ID list, ID map, and the adjacency list data. Otherwise pass through
        without further actions.

        Note:
            Does not raise error even if the node alrealy exists.

        """
        if node_id not in self._node_idmap:
            self._node_idmap[node_id] = self.num_nodes
            self.nodes.append(node_id)
            self._data.append({})

    def _add_edge_from_idx(self, idx1: int, idx2: int, weight: float):
        """Add an edge based on the head and tail node index with weight."""
        self._data[idx1][idx2] = weight
        self._num_edges += 1

    def add_edge(
        self,
        id1: str,
        id2: str,
        weight: float = 1.0,
        directed: bool = False,
    ):
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

    def read(
        self,
        path: str,
        weighted: bool,
        directed: bool,
        delimiter: str = "\t",
    ):
        """Read an edgelist file and create sparse graph.

        Note:
            Implicitly discard zero weighted edges; if the same edge is defined
            multiple times with different edge weights, then the last specified
            weight will be used (warning for such behavior will be printed).

        Args:
            path (str): path to edgelist file, where the file is tab
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
        with open(path, "r") as f:
            for edge_line in f:
                edge = self._read_edge_line(edge_line, weighted, delimiter)
                self.add_edge(*edge, directed)

    def save(self, path: str, unweighted: bool = False, delimiter: str = "\t"):
        """Save AdjLst as an ``.edg`` edge list file.

        Args:
            unweighted (bool): If set to True, only write two columns,
                corresponding to the head and tail nodes of the edges, and
                ignore the edge weights (default: :obj:`False`).
            delimiter (str): Delimiter for separating fields.

        """
        with open(path, "w") as f:
            for h, t, w in self.edges_iter:
                h_id, t_id = self.nodes[h], self.nodes[t]
                terms = (h_id, t_id) if unweighted else (h_id, t_id, str(w))
                f.write(f"{delimiter.join(terms)}\n")

    def to_csr(self) -> CSR:
        """Construct compressed sparse row matrix."""
        indptr = np.zeros(len(self.nodes) + 1, dtype=np.uint32)
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

    def to_dense(self) -> AdjMat:
        """Construct dense adjacency matrix.

        Note:
            This method does not return DenseGraph object, but instead return
            dense adjacency matrix as NDArray, the index is the same
            as that of ``nodes``.

        Return:
            NDArray: Full adjacency matrix as 2d numpy array.

        """
        n_nodes = len(self.nodes)
        mat = np.zeros((n_nodes, n_nodes))

        for src_node, src_nbrs in enumerate(self._data):
            for dst_node in src_nbrs:
                mat[src_node, dst_node] = src_nbrs[dst_node]

        return mat

    @classmethod
    def from_mat(cls, adj_mat: AdjMat, node_ids: List[str], **kwargs):
        """Construct graph using adjacency matrix and node IDs.

        Args:
            adj_mat(NDArray): 2D numpy array of adjacency matrix
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
        Read ``.edg`` file and create ``SparseGraph`` object using
        ``.read_edg`` method.

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
        self.data: Optional[Float32Array] = None
        self.indptr: Optional[Uint32Array] = None
        self.indices: Optional[Uint32Array] = None

    @property
    def num_edges(self) -> int:
        """Return the number of edges in the graph."""
        if self.indptr is not None:
            return self.indptr[-1]
        else:
            raise ValueError("Empty graph.")

    def read_edg(
        self,
        path: str,
        weighted: bool,
        directed: bool,
        delimiter: str = "\t",
    ):
        """Create CSR sparse graph from edge list.

        First create ``AdjlstGraph`` by reading the edge list file, and then
        convert to ``SparseGraph`` via ``to_csr``.

        Args:
            path (str): path to edgelist file.
            weighted (bool): whether the graph is weighted.
            directed (bool): whether the graph is directed.
            delimiter (str): delimiter used between node IDs.

        """
        g = AdjlstGraph()
        g.read(path, weighted, directed, delimiter)
        self.set_node_ids(g.nodes)
        self.indptr, self.indices, self.data = g.to_csr()

    def read_npz(self, path: str, weighted: bool, implicit_ids: bool = False):
        """Directly read a CSR sparse graph.

        Note:
            To generate a CSR file compatible with PecanPy, first load the graph
                as a sparse graph using the SparseGraph (with ``csr=True``).
                Then save the sparse graph to a csr file using the ``save``
                method from ``SparseGraph``. The saved ``.npz`` file can then
                be loaded directly by ``SparseGraph`` later.

        Args:
            path (str): path to the csr file, which is an npz file with four
                arrays with keys 'IDs', 'data', 'indptr', 'indices', which
                correspond to the node IDs, the edge weights, the offset array
                for each node, and the indices of the edges.
            weighted (bool): whether the graph is weighted, if unweighted,
                all edge weights will be converted to 1.
            directed (bool): not used, for compatibility with ``SparseGraph``.
            implicit_ids (bool): Implicitly set the node IDs to the canonical
                node ordering from the CSR graph. If unset and the `IDs` field
                is not found in the input CSR graph, a warning message will be
                displayed on screen. The missing `IDs` field can happen, for
                example, when the user uses the CSR graph prepared by
                `scipy.sparse.csr`.

        """
        raw = np.load(path)
        self.indptr = raw["indptr"].astype(np.uint32)
        self.indices = raw["indices"].astype(np.uint32)
        self.data = raw["data"].astype(np.float32)
        if self.data is None:
            raise ValueError("Adjacency matrix data not found.")
        elif not weighted:
            self.data[:] = 1.0  # overwrite edge weights with constant

        self.set_node_ids(
            raw.get("IDs"),
            implicit_ids=implicit_ids,
            num_nodes=int(self.indptr.size - 1),
        )

    def save(self, path: str):
        """Save CSR as ``.csr.npz`` file."""
        np.savez(
            path,
            IDs=self.nodes,
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
        g.set_node_ids(adjlst_graph.nodes)
        g.indptr, g.indices, g.data = adjlst_graph.to_csr()
        return g

    @classmethod
    def from_mat(cls, adj_mat: AdjMat, node_ids: List[str], **kwargs):
        """Construct csr graph using adjacency matrix and node IDs.

        Note:
            Only consider positive valued edges.

        Args:
            adj_mat(NDArray): 2D numpy array of adjacency matrix
            node_ids(:obj:`list` of str): node ID list

        """
        g = cls(**kwargs)
        g.set_node_ids(node_ids)
        adjlst_graph = AdjlstGraph.from_mat(adj_mat, node_ids)
        g.indptr, g.indices, g.data = adjlst_graph.to_csr()
        return g


class DenseGraph(BaseGraph):
    """Dense Graph object that stores graph as array.

    Examples:
        Read ``.npz`` files and create ``DenseGraph`` object using ``read_npz``

        >>> from pecanpy.graph import DenseGraph
        >>>
        >>> g = DenseGraph() # initialize DenseGraph object
        >>>
        >>> g.read_npz(paht_to_npz_file, weighted=True, directed=False)

        Read ``.edg`` files and create ``DenseGraph`` object using ``read_edg``

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
        self._data: Optional[AdjMat] = None
        self._nonzero: Optional[AdjNonZeroMat] = None

    @property
    def num_edges(self) -> int:
        """Return the number of edges in the graph."""
        if self.nonzero is not None:
            return self.nonzero.sum()
        else:
            raise ValueError("Empty graph.")

    @property
    def data(self) -> Optional[AdjMat]:
        """Return the adjacency matrix."""
        return self._data

    @data.setter
    def data(self, data: AdjMat):
        """Set adjacency matrix and the corresponding nonzero matrix."""
        self._data = data.astype(float)
        self._nonzero = np.array(self._data != 0, dtype=bool)

    @property
    def nonzero(self) -> Optional[AdjNonZeroMat]:
        """Return the nonzero mask for the adjacency matrix."""
        return self._nonzero

    def read_npz(self, path: str, weighted: bool, implicit_ids: bool = False):
        """Read ``.npz`` file and create dense graph.

        Args:
            path (str): path to ``.npz`` file.
            weighted (bool): whether the graph is weighted, if unweighted,
                all none zero weights will be converted to 1.
            implicit_ids (bool): Implicitly set the node IDs to the canonical
                ordering from the dense adjacency matrix object. If unset and
                the `IDs` field is not found in the object, a warning message
                will be displayed on screen. This warning message can be
                suppressed if `implicit_ids` is set to True as a confirmation
                of the behavior.

        """
        raw = np.load(path)
        self.data = raw["data"]
        if not weighted:  # overwrite edge weights with constant
            self.data = self.nonzero * 1.0  # type: ignore

        self.set_node_ids(
            raw.get("IDs"),
            implicit_ids=implicit_ids,
            num_nodes=self.data.shape[0],
        )

    def read_edg(
        self,
        path: str,
        weighted: bool,
        directed: bool,
        delimiter: str = "\t",
    ):
        """Read an edgelist file and construct dense graph."""
        g = AdjlstGraph()
        g.read(path, weighted, directed, delimiter)

        self.set_node_ids(g.nodes)
        self.data = g.to_dense()

    def save(self, path: str):
        """Save dense graph  as ``.dense.npz`` file."""
        np.savez(path, data=self.data, IDs=self.nodes)

    @classmethod
    def from_adjlst_graph(cls, adjlst_graph, **kwargs):
        """Construct dense graph from adjacency list graph.

        Args:
            adjlst_graph (:obj:`pecanpy.graph.AdjlstGraph`): Adjacency list
                graph to be converted.

        """
        g = cls(**kwargs)
        g.set_node_ids(adjlst_graph.nodes)
        g.data = adjlst_graph.to_dense()
        return g

    @classmethod
    def from_mat(cls, adj_mat: AdjMat, node_ids: List[str], **kwargs):
        """Construct dense graph using adjacency matrix and node IDs.

        Args:
            adj_mat(NDArray): 2D numpy array of adjacency matrix
            node_ids(:obj:`list` of str): node ID list

        """
        g = cls(**kwargs)
        g.data = adj_mat
        g.set_node_ids(node_ids)
        return g
