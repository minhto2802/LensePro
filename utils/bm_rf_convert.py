import numpy as np
from scipy import interpolate
import scipy.spatial.qhull as qhull

from utils.misc import timer


def BMtoRF(p, prb_radius, arc, depth, resol, method='cubic'):
    px_r = resol[2]
    px_th = resol[3]

    min_r = prb_radius
    max_r = prb_radius + depth

    max_y = prb_radius + depth
    nY = p.shape[0]
    nX = p.shape[1]
    maxX = nX / 2 * resol[0]
    minX = -maxX

    max_theta = arc / prb_radius / 2
    min_theta = -max_theta

    rc = np.arange(min_r, max_r, px_r)
    thc = np.arange(min_theta, max_theta, px_th)

    min_y = prb_radius * np.cos(max_theta)

    x = np.arange(minX, maxX, resol[0])
    y = np.linspace(min_y, max_y, nY)

    xSR, ySR = np.meshgrid(x, y)

    rSR, tSR = np.meshgrid(rc, thc)

    sin = np.sin(tSR)
    cos = np.cos(tSR)
    px = rSR * sin
    py = rSR * cos

    new_grid = interpolate.griddata((xSR.flatten(), ySR.flatten()), p.flatten(), (px, py), method=method)
    q = new_grid.transpose()

    return q


def RFtoBM(p, prb_radius, arc, depth, resol, method='cubic'):
    px_r = resol[2]
    px_th = resol[3]

    nR = p.shape[0]
    nTheta = p.shape[1]

    min_r = prb_radius
    max_r = px_r * (nR - 1) + prb_radius

    max_theta = (nTheta - 1 - nTheta / 2) * px_th
    min_theta = -nTheta * px_th / 2

    rc = np.linspace(min_r, max_r, nR)
    thc = np.linspace(min_theta, max_theta, nTheta)

    nY = 616
    nX = 756
    maxX = nX / 2 * resol[0]
    minX = -maxX
    max_y = prb_radius + depth
    min_y = prb_radius * np.cos(max_theta)

    x = np.arange(minX, maxX, resol[0])
    y = np.linspace(min_y, max_y, nY)

    xSR, ySR = np.meshgrid(x, y)

    rSR, tSR = np.meshgrid(rc, thc)

    pth = np.arctan(xSR / ySR)
    pr = np.sqrt(np.power(xSR, 2) + np.power(ySR, 2))

    # new_grid = interpolate.griddata((rSR.flatten(), tSR.flatten()), p.transpose().flatten(), (pr, pth), method=method)
    new_grid = grid_data2(rSR, tSR, pr, pth, p, method=method)
    # show_2d(new_grid)

    return new_grid


def show_2d(img):
    import pylab as plt
    import matplotlib
    matplotlib.use('TkAgg')
    plt.imshow(img, cmap='gray')
    plt.show()


@timer
def grid_data(r_sr, t_sr, pr, pth, p, method='linear'):
    vtx, wts = interp_weights(
        np.vstack((r_sr.flatten(), t_sr.flatten())).T,
        np.vstack((pr.flatten(), pth.flatten())).T)
    return interp(p.transpose().flatten(), vtx, wts).reshape(pr.shape)


def grid_data2(r_sr, t_sr, pr, pth, p, method='linear'):
    new_grid = interpolate.griddata(
        (r_sr.flatten(), t_sr.flatten()), p.transpose().flatten(), (pr, pth), method=method)
    return new_grid


def interp_weights(xyz, uvw, d=2):
    tri = qhull.Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))


def interp(values, vtx, wts):
    return np.einsum('nj,nj->n', np.take(values, vtx), wts)
