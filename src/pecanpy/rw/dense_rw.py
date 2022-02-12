"""Dense Graph object equipped with random walk computation."""
import numpy as np
from numba import njit

from ..graph import DenseGraph


class DenseRWGraph(DenseGraph):
    """Dense Graph object equipped with random walk computation."""

    def get_noise_thresholds(self):
        """Compute average edge weights."""
        noise_threshold_ary = np.zeros(self.num_nodes, dtype=np.float32)
        for i in range(self.num_nodes):
            weights = self.data[i, self.nonzero[i]]
            noise_threshold_ary[i] = weights.mean() + self.gamma * weights.std()
        noise_threshold_ary = np.maximum(noise_threshold_ary, 0)

        return noise_threshold_ary

    def get_has_nbrs(self):
        """Wrap ``has_nbrs``."""
        nonzero = self.nonzero

        @njit(nogil=True)
        def has_nbrs(idx):
            for j in range(nonzero.shape[1]):
                if nonzero[idx, j]:
                    return True
            return False

        return has_nbrs

    @staticmethod
    @njit(nogil=True)
    def get_normalized_probs(
        data,
        nonzero,
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
        nbrs_ind = nonzero[cur_idx]
        unnormalized_probs = data[cur_idx].copy()

        if prev_idx is not None:  # 2nd order biased walks
            non_com_nbr = np.logical_and(nbrs_ind, ~nonzero[prev_idx])
            non_com_nbr[prev_idx] = False  # exclude previous state from out biases

            unnormalized_probs[non_com_nbr] /= q  # apply out biases
            unnormalized_probs[prev_idx] /= p  # apply the return bias

        unnormalized_probs = unnormalized_probs[nbrs_ind]
        normalized_probs = unnormalized_probs / unnormalized_probs.sum()

        return normalized_probs

    @staticmethod
    @njit(nogil=True)
    def get_extended_normalized_probs(
        data,
        nonzero,
        p,
        q,
        cur_idx,
        prev_idx,
        noise_threshold_ary,
    ):
        """Calculate node2vec+ transition probabilities."""
        cur_nbrs_ind = nonzero[cur_idx]
        unnormalized_probs = data[cur_idx].copy()

        if prev_idx is not None:  # 2nd order biased walks
            prev_nbrs_weight = data[prev_idx].copy()

            # Note: we assume here the network is undirectly, hence the edge
            # weight connecting the next to prev is the same as the reverse.
            out_ind = cur_nbrs_ind & (prev_nbrs_weight < noise_threshold_ary)
            out_ind[prev_idx] = False  # exclude previous state from out biases

            # print("CURRENT: ", cur_idx)
            # print("INOUT: ", np.where(out_ind)[0])
            # print("NUM INOUT: ", out_ind.sum(), "\n")

            t = prev_nbrs_weight[out_ind] / noise_threshold_ary[out_ind]
            # optional nonlinear parameterization
            # b = 1; t = b * t / (1 - (b - 1) * t)

            # compute out biases
            alpha = 1 / q + (1 - 1 / q) * t

            # suppress noisy edges
            alpha[
                unnormalized_probs[out_ind] < noise_threshold_ary[cur_idx]
            ] = np.minimum(1, 1 / q)
            unnormalized_probs[out_ind] *= alpha  # apply out biases
            unnormalized_probs[prev_idx] /= p  # apply the return bias

        unnormalized_probs = unnormalized_probs[cur_nbrs_ind]
        normalized_probs = unnormalized_probs / unnormalized_probs.sum()

        return normalized_probs
