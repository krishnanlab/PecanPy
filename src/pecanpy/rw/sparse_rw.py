"""Sparse Graph equipped with random walk computation."""
import numpy as np
from numba import boolean
from numba import njit

from ..graph import SparseGraph


class SparseRWGraph(SparseGraph):
    """Sparse Graph equipped with random walk computation."""

    def get_has_nbrs(self):
        """Wrap ``has_nbrs``."""
        indptr = self.indptr

        @njit(nogil=True)
        def has_nbrs(idx):
            return indptr[idx] != indptr[idx + 1]

        return has_nbrs

    def get_noise_thresholds(self):
        """Compute average edge weights."""
        data = self.data
        indptr = self.indptr

        noise_threshold_ary = np.zeros(self.num_nodes, dtype=np.float32)
        for i in range(self.num_nodes):
            noise_threshold_ary[i] = (
                data[indptr[i] : indptr[i + 1]].mean()
                + self.gamma * data[indptr[i] : indptr[i + 1]].std()
            )
        noise_threshold_ary = np.maximum(noise_threshold_ary, 0)

        return noise_threshold_ary

    @staticmethod
    @njit(nogil=True)
    def get_normalized_probs_first_order(data, indices, indptr, cur_idx):
        """Clculate first order transition probabilities.

        Note:
            This function does NOT check whether p = q = 1, which is the
            requried setup for first order random walk. Need to check before
            calling this function.

        """
        _, unnormalized_probs = get_nbrs(indptr, indices, data, cur_idx)
        return unnormalized_probs / unnormalized_probs.sum()

    @staticmethod
    @njit(nogil=True)
    def get_normalized_probs(
        data,
        indices,
        indptr,
        p,
        q,
        cur_idx,
        prev_idx,
        noise_threshold_ary,
    ):
        """Calculate node2vec transition probabilities.

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
        nbrs_idx, unnormalized_probs = get_nbrs(indptr, indices, data, cur_idx)
        if prev_idx is not None:  # 2nd order biased walk
            prev_ptr = np.where(nbrs_idx == prev_idx)[0]
            src_nbrs_idx, src_nbrs_wts = get_nbrs(indptr, indices, data, prev_idx)

            # Neighbors of current but not previous
            non_com_nbr = isnotin(nbrs_idx, src_nbrs_idx)
            non_com_nbr[prev_ptr] = False  # exclude prev state from out biases

            unnormalized_probs[non_com_nbr] /= q  # apply out biases
            unnormalized_probs[prev_ptr] /= p  # apply the return bias

        normalized_probs = unnormalized_probs / unnormalized_probs.sum()

        return normalized_probs

    @staticmethod
    @njit(nogil=True)
    def get_extended_normalized_probs(
        data,
        indices,
        indptr,
        p,
        q,
        cur_idx,
        prev_idx,
        noise_threshold_ary,
    ):
        """Calculate node2vec+ transition probabilities."""
        nbrs_idx, unnormalized_probs = get_nbrs(indptr, indices, data, cur_idx)
        if prev_idx is not None:  # 2nd order biased walk
            prev_ptr = np.where(nbrs_idx == prev_idx)[0]
            src_nbrs_idx, src_nbrs_wts = get_nbrs(indptr, indices, data, prev_idx)
            out_ind, t = isnotin_extended(
                nbrs_idx,
                src_nbrs_idx,
                src_nbrs_wts,
                noise_threshold_ary,
            )  # determine out edges
            out_ind[prev_ptr] = False  # exclude prevstate from out biases

            # compute out biases
            alpha = 1 / q + (1 - 1 / q) * t[out_ind]

            # surpress noisy edges
            alpha[
                unnormalized_probs[out_ind] < noise_threshold_ary[cur_idx]
            ] = np.minimum(1, 1 / q)
            unnormalized_probs[out_ind] *= alpha  # apply out biases
            unnormalized_probs[prev_ptr] /= p  # apply the return bias

        normalized_probs = unnormalized_probs / unnormalized_probs.sum()

        return normalized_probs


@njit(nogil=True)
def get_nbrs(indptr, indices, data, idx):
    """Return neighbor indices and weights of a specific node index."""
    start_idx, end_idx = indptr[idx], indptr[idx + 1]
    nbrs_idx = indices[start_idx:end_idx]
    nbrs_wts = data[start_idx:end_idx].copy()
    return nbrs_idx, nbrs_wts


@njit(nogil=True)
def isnotin(ptr_ary1, ptr_ary2):
    """Find node2vec out edges.

    The node2vec out edges is determined by non-common neighbors. This function
    find out neighbors of node1 that are not neighbors of node2, by picking out
    values in ``ptr_ary1`` but not in ``ptr_ary2``, which correspond to the
    neighbor pointers for the current state and the previous state, resp.

    Note:
        This function does not remove the index of the previous state. Instead,
    the index of the previous state will be removed once the indicator is
    returned to the ``get_normalized_probs``.

    Args:
        ptr_ary1 (Uint32Array): array of pointers to
            the neighbors of the current state
        ptr_ary2 (Uint32Array): array of pointers to
            the neighbors of the previous state

    Returns:
        Indicator of whether a neighbor of the current state is considered as
            an "out edge"

    Example:
        The values in the two neighbor pointer arrays are sorted ascendingly.
        The main idea is to scan through ``ptr_ary1`` and compare the values in
        ``ptr_ary2``. In this way, at most one pass per array is needed to find
        out the non-common neighbor pointers instead of a nested loop (for each
        element in ``ptr_ary1``, compare against every element in``ptr_ary2``),
        which is much slower. Checkout the following example for more intuition.
        The ``*`` above ``ptr_ary1`` and ``ptr_ary2`` indicate the indices
        ``idx1`` and ``idx2``, respectively, which keep track of the scaning
        progress.

        >>> ptr_ary1 = [1, 2, 5]
        >>> ptr_ary2 = [1, 5]
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

    """
    indicator = np.ones(ptr_ary1.size, dtype=boolean)
    idx2 = 0
    for idx1 in range(ptr_ary1.size):
        if idx2 == ptr_ary2.size:  # end of ary2
            break

        ptr1 = ptr_ary1[idx1]
        ptr2 = ptr_ary2[idx2]

        if ptr1 < ptr2:
            continue

        elif ptr1 == ptr2:  # found a matching value
            indicator[idx1] = False
            idx2 += 1

        elif ptr1 > ptr2:
            # sweep through ptr_ary2 until ptr2 catch up on ptr1
            for j in range(idx2, ptr_ary2.size):
                ptr2 = ptr_ary2[j]
                if ptr2 == ptr1:
                    indicator[idx1] = False
                    idx2 = j + 1
                    break

                elif ptr2 > ptr1:
                    idx2 = j
                    break

    return indicator


@njit(nogil=True)
def isnotin_extended(ptr_ary1, ptr_ary2, wts_ary2, noise_thresholds):
    """Find node2vec+ out edges.

    The node2vec+ out edges is determined by considering the edge weights
    connecting node2 (the potential next state) to the previous state. Unlinke
    node2vec, which only considers neighbors of current state that are not
    neighbors of the previous state, node2vec+ also considers neighbors of
    the previous state as out edges if the edge weight is below average.

    Args:
        ptr_ary1 (Uint32Array): array of pointers to the neighbors of the
            current state
        ptr_ary2 (Uint32Array): array of pointers to the neighbors of the
            previous state
        wts_ary2 (Float32Array): array of edge weights of the previous state
        noise_thresholds (Float32Array): array of noisy edge threshold computed
            based on the average and the std of the edge weights of each node

    Return:
        Indicator of whether a neighbor of the current state is considered as
            an "out edge", with the corresponding parameters used to fine tune
            the out biases

    """
    indicator = np.ones(ptr_ary1.size, dtype=boolean)
    t = np.zeros(ptr_ary1.size, dtype=np.float32)
    idx2 = 0
    for idx1 in range(ptr_ary1.size):
        if idx2 >= ptr_ary2.size:  # end of ary2
            break

        ptr1 = ptr_ary1[idx1]
        ptr2 = ptr_ary2[idx2]

        if ptr1 < ptr2:
            continue

        elif ptr1 == ptr2:  # found a matching value
            # If connection is not loose, identify as an in-edge
            if wts_ary2[idx2] >= noise_thresholds[ptr2]:
                indicator[idx1] = False
            else:
                t[idx1] = wts_ary2[idx2] / noise_thresholds[ptr2]
            idx2 += 1

        elif ptr1 > ptr2:
            # Sweep through ptr_ary2 until ptr2 catch up on ptr1
            for j in range(idx2 + 1, ptr_ary2.size):
                ptr2 = ptr_ary2[j]
                if ptr2 == ptr1:
                    if wts_ary2[j] >= noise_thresholds[ptr2]:
                        indicator[idx1] = False
                    else:
                        t[idx1] = wts_ary2[j] / noise_thresholds[ptr2]
                    idx2 = j + 1
                    break

                elif ptr2 > ptr1:
                    idx2 = j
                    break

    return indicator, t
