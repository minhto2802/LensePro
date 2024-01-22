import numpy as np
from scipy.spatial.distance import pdist, squareform


def get_neighbors(coors, core_id, ts_cid_to_id: dict, n_neighbors=16):
    """

    :param coors:
    :param core_id:
    :param ts_cid_to_id:
    :param n_neighbors:
    :return:
    """
    all_core_id = list(ts_cid_to_id.keys())
    neighbors = np.zeros((coors.shape[0], n_neighbors))
    import time
    tic = time.time()
    for cid in all_core_id:
        print(cid)
        coor_cid = coors[core_id == cid]
        neighbors[ts_cid_to_id[cid]] = get_neighbors_single_core(coor_cid,
                                                                 cid,
                                                                 ts_cid_to_id,
                                                                 n_neighbors)
    print(time.time() - tic)
    exit()


def get_neighbors_single_core(coor_cid, cid, ts_cid_to_id, n_neighbors=16):
    """

    :param cid:
    :param coor_cid: coor of a single core
    :param ts_cid_to_id:
    :param n_neighbors:
    :return:
    """
    current_id = ts_cid_to_id[cid]
    n = coor_cid.shape[0]
    dist = squareform(pdist(coor_cid))
    neighbors = np.zeros((n, n_neighbors), dtype=int)
    for i in range(n):
        neighbors[i] = current_id[np.argsort(dist[i])[1:n_neighbors + 1]]
    return neighbors


def get_idx(i, m):
    # i: index of target location
    # m: len(coor_cid)
    j = np.arange(i + 1, m)
    idx = m * i + j - ((i + 2) * (i + 1)) // 2
    return idx
