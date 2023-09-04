"""Experimental features."""
import numpy as np
from numba import njit
from pecanpy.pecanpy import Base
from pecanpy.rw.dense_rw import DenseRWGraph


class Node2vecPlusPlus(Base, DenseRWGraph):
    """Continuous extension of node2vec+ with DenseOTF framework.

    In node2vec+ (see `DenseRWGraph.get_extended_normalized_probs`), there is
    discontinuous region of the bias-factor (alpha). More specifically, the
    transition between the noisy-edge region (w1 < 1 and w2 < 1, where w1 is
    the normalized edge weight connecting from current to the previous node,
    and w2 is similarly defined for the edge weight connecting from the next
    to the previous node), and the "in-out" region (w1 > 1 or w2 > 1).

    This continuous extension version of node2vec+, i.e., node2vec++, aims to
    provide continuity to those regions by parameterizing the bias-factor as
    a continuous function of w1 and w2. The basic idea is to use w2 to control
    the interpolation between 1 and 1 / q as before, but in addition, use w1
    to parameterize the curvature of the interpolation, so as w1 approaches
    zero, the bias-factor goes to min{1, 1 / q} (note that previously, the
    bias-factor is set to min{1, 1 / q} whenever w1 falls below one).

    """

    def __init__(self, *args, **kwargs):
        Base.__init__(self, *args, **kwargs)

    def get_move_forward(self):
        """Wrap ``move_forward``."""
        data = self.data
        nonzero = self.nonzero
        p = self.p
        q = self.q

        noise_thresholds = self.get_noise_thresholds()
        get_normalized_probs = self.get_normalized_probs

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
        """Calculate node2vec++ transition probabilities."""
        cur_nbrs_ind = nonzero[cur_idx]
        cur_nbrs_weight = data[cur_idx].copy()

        if prev_idx is not None:  # 2nd order biased walks
            prev_nbrs_weight = data[prev_idx].copy()

            # Note: we assume here the network is undirected, hence the edge
            # weight connecting the next to prev is the same as the reverse.
            out_ind = cur_nbrs_ind & (prev_nbrs_weight < noise_threshold_ary)
            out_ind[prev_idx] = False  # exclude previous state from out biases

            t = prev_nbrs_weight[out_ind] / noise_threshold_ary[out_ind]
            # Determine whether to use '1 - t' or 't' depending on whether q
            # is less than or greater than one so that alpha is suppressed to
            # min{1, 1 / q} as w1 approaches 0.
            t = 1 - t.clip(0, 1) if q < 1 else t.clip(0, 1)
            b = cur_nbrs_weight[out_ind] / noise_threshold_ary[out_ind]

            # compute out biases
            scale = np.abs(1 - 1 / q)
            offset = np.minimum(1, 1 / q)
            alpha = t * b / (1 + (b - 1)) * scale + offset

            cur_nbrs_weight[out_ind] *= alpha  # apply out biases
            cur_nbrs_weight[prev_idx] /= p  # apply the return bias

        unnormalized_probs = cur_nbrs_weight[cur_nbrs_ind]
        normalized_probs = unnormalized_probs / unnormalized_probs.sum()

        return normalized_probs
