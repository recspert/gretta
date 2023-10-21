from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from numba import njit, prange
from numba.typed import List


def arrange_indices(idx, mode_mask=None, shape=None):
    n = idx.shape[1]
    res = [[]] * n
    if mode_mask is None:
        mode_mask = [True] * n
    sizes = list(shape) if shape else [None]*n

    n_active_modes = sum(mode_mask)
    if n_active_modes == 0:
        return res

    if n_active_modes == 1:
        mode = mode_mask.index(True)
        res[mode] = arrange_index(idx[:, mode], size=sizes[mode])
        return res

    with ThreadPoolExecutor(max_workers=n_active_modes) as executor:
        arranged_futures = {
            executor.submit(arrange_index, idx[:, mode], size=sizes[mode]): mode
            for mode in range(n) if mode_mask[mode]
        }
        for future in as_completed(arranged_futures):
            mode = arranged_futures[future]
            res[mode] = future.result()
    return res

# numba at least up to v0.50.1 only supports the 1st argument of np.unique
# https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html
def arrange_index(array, typed=True, size=None):
    '''Mainly used in Tucker decomposition calculations. Enables parallelism.
    '''
    unqs, unq_inv, unq_cnt = np.unique(array, return_inverse=True, return_counts=True)
    inds = np.split(np.argsort(unq_inv), np.cumsum(unq_cnt[:-1]))

    if typed: # reflected lists are being deprecated in numba - switch to typed lists by default
        inds_typed = List()
        for ind in inds:
            inds_typed.append(ind)
        inds = inds_typed

    if (size is not None) and (len(unqs) < size):
        unqs, inds = fill_missing_sorted(unqs, inds, size)
        unqs = np.array(unqs)
    return unqs, inds

@njit
def fill_missing_sorted(arr, inds, size):
    filler = inds[0][0:0]
    arr_filled = np.empty(size, dtype=arr.dtype)
    inds_filled = List()
    pos = 0
    for i in range(size):
        if i == arr[pos]:
            inds_filled.append(inds[pos])
            pos += 1
        else:
            inds_filled.append(filler)
        arr_filled[i] = i
    return arr_filled, inds_filled


def ttm3d_seq(idx, val, shape, U, V, modes, dtype=None):
    mode1, mat_mode1 = modes[0]
    mode2, mat_mode2 = modes[1]

    u = U.T if mat_mode1 == 1 else U
    v = V.T if mat_mode2 == 1 else V

    mode0, = [x for x in (0, 1, 2) if x not in (mode1, mode2)]
    new_shape = (shape[mode0], U.shape[1-mat_mode1], V.shape[1-mat_mode2])

    res = np.zeros(new_shape, dtype=dtype)
    dttm_seq(idx, val, u, v, mode0, mode1, mode2, res)
    return res


def ttm3d_par(idx, val, shape, U, V, modes, unqs, inds, dtype=None):
    mode1, mat_mode1 = modes[0]
    mode2, mat_mode2 = modes[1]

    u = U.T if mat_mode1 == 1 else U
    v = V.T if mat_mode2 == 1 else V

    mode0, = [x for x in (0, 1, 2) if x not in (mode1, mode2)]
    new_shape = (shape[mode0], U.shape[1-mat_mode1], V.shape[1-mat_mode2])

    res = np.zeros(new_shape, dtype=dtype)
    dttm_par(idx, val, u, v, mode1, mode2, unqs, inds, res)
    return res


@njit(nogil=True)
def dttm_seq(idx, val, u, v, mode0, mode1, mode2, res):
    new_shape1 = u.shape[1]
    new_shape2 = v.shape[1]
    for i in range(len(val)):
        i0 = idx[i, mode0]
        i1 = idx[i, mode1]
        i2 = idx[i, mode2]
        vv = val[i]
        for j in range(new_shape1):
            uij = u[i1, j]
            for k in range(new_shape2):
                vik = v[i2, k]
                res[i0, j, k] += vv * uij * vik


@njit(parallel=True)
def dttm_par(idx, val, mat1, mat2, mode1, mode2, unqs, inds, res):
    r1 = mat1.shape[1]
    r2 = mat2.shape[1]
    n = len(unqs)

    for s in prange(n):
        i0 = unqs[s]
        ul = inds[s]
        for pos in ul:
            i1 = idx[pos, mode1]
            i2 = idx[pos, mode2]
            vp = val[pos]
            for j1 in range(r1):
                for j2 in range(r2):
                    res[i0, j1, j2] += vp * mat1[i1, j1] * mat2[i2, j2]
