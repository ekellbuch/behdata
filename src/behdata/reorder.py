from networkx.utils import cuthill_mckee_ordering
import networkx as nx
import numpy as np
from scipy.cluster import hierarchy


def cuthill_mckee(C, return_order=False):

    G = nx.Graph(C)
    rorder = list(cuthill_mckee_ordering(G))

    if return_order:
        return C[np.ix_(rorder, rorder)], rorder
    else:
        return C[np.ix_(rorder, rorder)]


def optimal_overleaf(C, return_order=False):
    """
    we are ordering the vertices by their neighborhood similarities
    ordering is consistent with a hierarchichal binary tree
    ordering is computed globally so as to minimize the sum of distances between
    successive rows while traversing the binary tree in a depth-first-order

    Finds a linkage matrix Z, and reorder the cut tree (forest).


    step 1: (hierarchy ward)
    find a condensed matrix that stores the pairwise distances of the observations.
    here we are using the ward distance (takes into account cardinality.

    step 2: (optimal_leaf_ordering)
    Compute the optimal leaf order for Z (according to C)
    and return an optimally sorted Z.

    Args:
        C:
        return_order:

    Returns:

    """
    Z = hierarchy.ward(C)
    rorder = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z, C))
    if return_order:
        return C[np.ix_(rorder, rorder)], rorder
    else:
        return C[np.ix_(rorder, rorder)]


def closed_sum(C_pr, axis=0, return_order=False):
    # should not update 1, else also need to change expand dims
    # must sum 1 at
    assert np.allclose(C_pr.sum(axis), 1)
    k1, k2 = C_pr.shape

    avg_ind = np.sum(C_pr * np.arange(k1)[:, None], axis=axis)
    perm = np.argsort(avg_ind)
    ends_ = (avg_ind == 0).sum()
    new_l = perm.copy()
    new_l[: k2 - ends_] = perm[ends_:]
    new_l[k2 - ends_:] = perm[:ends_]

    if return_order:
        return C_pr[:, new_l], new_l
    else:
        return C_pr[:, new_l]
