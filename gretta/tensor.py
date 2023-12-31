import warnings
from functools import wraps

import numpy as np
from scipy.sparse.linalg import svds

try:
    from sklearn.utils.extmath import randomized_svd
except ImportError:
    randomized_svd = None

from .sparse import arrange_indices, ttm3d_seq, ttm3d_par


class SATFError(Exception):
    pass


def suppress(warning):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=warning)
                res = function(*args, **kwargs)
            return res
        return wrapper
    return decorator

def initialize_columnwise_orthonormal(dims, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    u = rng.standard_normal(dims)
    u, _ = np.linalg.qr(u, mode='reduced')
    return u

def core_growth_callback(growth_tol):
    def check_core_growth(step, core_norm, factors):
        g_growth = (core_norm - check_core_growth.core_norm) / core_norm
        check_core_growth.core_norm = core_norm
        print(f'growth of the core: {g_growth}')
        if g_growth < growth_tol:
            print(f'Core is no longer growing. Norm of the core: {core_norm}.')
            raise StopIteration
    check_core_growth.core_norm = 0
    return check_core_growth

def tensor_factors(
        idx, val, shape, mlrank,
        attention_matrix = None,
        scaling_weights = None,
        max_iters = 20,
        parallel_ttm = False,
        growth_tol = 0.001,
        randomized=True,
        seed = None,
        iter_callback=None,
    ):
    _, n_items, n_positions = shape
    r0, r1, r2 = mlrank

    tensor_data = idx, val, shape
    if not isinstance(parallel_ttm, (list, tuple)):
        parallel_ttm = [parallel_ttm] * len(shape)

    assert len(shape) == len(parallel_ttm)

    index_data = arrange_indices(idx, parallel_ttm)
    ttm = [ttm3d_par if par else ttm3d_seq for par in parallel_ttm]

    random_state = np.random if seed is None else np.random.RandomState(seed)
    u1 = initialize_columnwise_orthonormal((n_items, r1), random_state)
    uw = u1 # * scaling_weights[:, np.newaxis]
    u2 = initialize_columnwise_orthonormal((n_positions, r2), random_state)
    ua = u2 # attention_matrix.dot(u2)

    if randomized:
        svd = randomized_svd
        svd_config = lambda rank: dict(n_components=rank, random_state=random_state)
    else:
        svd = svds
        svd_config = lambda rank: dict(k=rank, return_singular_vectors='u')

    if iter_callback is None:
        iter_callback = core_growth_callback(growth_tol)


    for step in range(max_iters):
        ttm0 = ttm[0](*tensor_data, ua, uw, ((2, 0), (1, 0)), *index_data[0]).reshape(shape[0], r1*r2)
        u0, *_ = svd(ttm0, **svd_config(r0))

        ttm1 = ttm[1](*tensor_data, ua, u0, ((2, 0), (0, 0)), *index_data[1]).reshape(shape[1], r0*r2)
        u1, *_ = svd(ttm1, **svd_config(r1))
        uw = u1 # * scaling_weights[:, np.newaxis]

        ttm2 = ttm[2](*tensor_data, uw, u0, ((1, 0), (0, 0)), *index_data[2]).reshape(shape[2], r0*r1)
        u2, ss, _ = svd(ttm2, **svd_config(r2))
        ua = u2 # attention_matrix.dot(u2)

        factors = (u0, u1, u2)
        try:
            iter_callback(step, np.linalg.norm(ss), factors)
        except StopIteration:
            break
    return factors


def get_scaling_weights(frequencies, scaling=1.0):
    return np.power(frequencies, 0.5*(scaling-1.0), where=frequencies>0)


def valid_mlrank(mlrank):
    prod = np.prod(mlrank)
    return all(prod//r > r for r in mlrank)
