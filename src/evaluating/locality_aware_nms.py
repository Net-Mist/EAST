import numpy as np
from shapely.geometry import Polygon
import time


def intersection(g, p):
    """
    Compute the IoU (Intersection over Union) of g and p
    Args:
        g: An array of shape 9. first 8 coordinates
        p: An array of shape 9. first 8 coordinates

    Returns:

    """
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = g.intersection(p).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter / union


def weighted_merge(g, p):
    g[:8] = (g[8] * g[:8] + p[8] * p[:8]) / (g[8] + p[8])
    g[8] = g[8] + p[8]
    return g


def standard_nms(S, thres):
    order = np.argsort(S[:, 8])[::-1]  # List of the ids of elements of S in decreasing order of score not yet processed
    keep = []  # List of ids to keep
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = np.array([intersection(S[i], S[t]) for t in order[1:]])

        inds = np.where(ovr <= thres)[0]
        order = order[inds + 1]

    return S[keep]


def nms_locality(polys, thres=0.3):
    """
    locality aware nms of EAST
    Args:
        polys: a N*9 numpy array. first 8 coordinates, then prob
        thres: merging threshold
    Returns: boxes after nms
    """
    S = []
    p = None
    for g in polys:
        if p is not None and intersection(g, p) > thres:
            p = weighted_merge(g, p)
        else:
            if p is not None:
                S.append(p)
            p = g
    if p is not None:
        S.append(p)

    if len(S) == 0:
        return np.array([])
    return standard_nms(np.array(S), thres)


if __name__ == '__main__':
    # 343,350,448,135,474,143,369,359
    print(Polygon(np.array([[343, 350], [448, 135],
                            [474, 143], [369, 359]])).area)
