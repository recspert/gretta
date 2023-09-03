from itertools import takewhile, count, islice

import numpy as np
from scipy.signal import fftconvolve
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator, svds
from numba import njit, prange

from polara.lib.sparse import arrange_indices

from tensor import core_growth_callback, initialize_columnwise_orthonormal

PARALLEL_MATVECS = True
FASTMATH_MATVECS = True
DTYPE = np.float64

class TESSAError(Exception):
    pass


def hankel_series_compress_2d(left_factors, right_factors):
    hankel_compressed = fftconvolve(left_factors[:, np.newaxis, :], right_factors[:, :, np.newaxis])
    return hankel_compressed.reshape(hankel_compressed.shape[0], -1)


@njit(parallel=False)
def hankel_series_matvec1(pos, vec, dim1, dim2):
    res = np.empty(dim1, dtype='f8')
    for i in range(dim1):
        j = pos - i
        val = 0.
        if j >= 0:
            if j < dim2:
                val = vec[j]
        res[i] = val
    return res


def accum_kron_weights(arranged_position_index, idx, entity1_factors, entity2_factors):
    e1_rank = entity1_factors.shape[1]
    e2_rank = entity2_factors.shape[1]
    n_points = len(arranged_position_index)
    res = np.empty((n_points, e2_rank, e1_rank), dtype=np.float64)
    for n, idx_rows in enumerate(arranged_position_index):
        e1_idx = idx[idx_rows, 0]
        e2_idx = idx[idx_rows, 1]
        w1 = entity1_factors[e1_idx, :]
        w2 = entity2_factors[e2_idx, :]
        accum_w_i = w2.T @ w1
        res[n, ...] = accum_w_i
    return res.reshape(n_points, e2_rank*e1_rank)


def series_matvec(factors, series_key, attention=None):
    series = ['attention', 'sequences']
    other_series = series[1 - series.index(series_key)]
    sf_dim, sf_rank = factors[other_series].shape
    n_pos, ee_rank = factors['accum_w_i'].shape
    dim_2 = sf_dim
    dim_1 = n_pos - dim_2 + 1
    
    @njit(parallel=PARALLEL_MATVECS, fastmath=FASTMATH_MATVECS)
    def hankel_series_mul(W, V, accum_w_i, n_pos, dim_1, dim_2):
        res = np.empty((n_pos, dim_1), dtype=np.float64)
        for i in prange(n_pos):
            tmp = np.dot(V, accum_w_i[i])
            vec = np.dot(W, tmp)
            res[i, :] = hankel_series_matvec1(i, vec, dim_1, dim_2)
        return res.sum(axis=0)
    
    def matvec(vec):
        series_factors = factors[other_series]
        accum_w_i = factors['accum_w_i']
        V = vec.reshape(ee_rank, sf_rank).T
        res = hankel_series_mul(series_factors, V, accum_w_i, n_pos, dim_1, dim_2)
        if attention is not None:
            res = attention.T.dot(res)
        return res
    return matvec


def series_rmatvec(factors, series_key, attention=None):
    series = ['attention', 'sequences']
    other_series = series[1 - series.index(series_key)]
    def rmatvec(vec):
        series_factors = factors[other_series]
        accum_w_i = factors['accum_w_i']
        if attention is not None:
            vec = attention.dot(vec)
        hankel_weights = fftconvolve(series_factors, vec[:, np.newaxis])
        res = accum_w_i.T @ hankel_weights
        return res.ravel()
    return rmatvec


@njit(parallel=PARALLEL_MATVECS, fastmath=FASTMATH_MATVECS)
def entity_fiber_matvec(arranged_entity_index, idx, entity_mode, other_factors, hankel_weights, vec):
    _, o_rank = other_factors.shape
    _, s_rank = hankel_weights.shape
    V = vec.reshape(o_rank, s_rank) # kron(b, a) x = a^T X b
    VX_cache = np.dot(hankel_weights, V.T)

    other_entity_mode = 1 - entity_mode
    n_entities = len(arranged_entity_index)
    res = np.empty(n_entities, dtype=np.float64)
    for main_entity_id in prange(n_entities):
        idx_rows = arranged_entity_index[main_entity_id]
        tmp = 0.
        for row_id in idx_rows:
            pos = idx[row_id, 2]
            other_entity_id = idx[row_id, other_entity_mode]
            w_i = other_factors[other_entity_id, :]
            tmp += np.dot(VX_cache[pos, :], w_i) # kron(b, a) x = a^T X b
        res[main_entity_id] = tmp
    return res

def entity_matvec(arranged_entity_index, idx, factors, entity_key, scaling=None):
    entities = ['users', 'items']
    entity_mode = entities.index(entity_key)
    other_entity = entities[1-entity_mode]
    def matvec(vec):
        other_factors = factors[other_entity]
        hankel_weights = factors['hankel_weights']
        result = entity_fiber_matvec(arranged_entity_index, idx, entity_mode, other_factors, hankel_weights, vec)
        if scaling is not None:
            result = scaling * result
        return result
    return matvec


@njit(parallel=PARALLEL_MATVECS, fastmath=FASTMATH_MATVECS)
def entity_fiber_rmatvec(arranged_position_index, idx, entity_mode, other_factors, hankel_weights, vec):
    _, o_rank = other_factors.shape
    n_pos = len(arranged_position_index)
    other_entity_mode = 1 - entity_mode
    accum_wev = np.empty((n_pos, o_rank), dtype=np.float64)
    for i in prange(n_pos):
        idx_rows = arranged_position_index[i]
        wev = np.zeros(o_rank, dtype=np.float64)
        for row_id in idx_rows:
            entities = idx[row_id, :2]
            main_entity = entities[entity_mode]
            other_entity = entities[other_entity_mode]
            val = vec[main_entity]
            w_i = other_factors[other_entity, :]
            wev += w_i * val
        accum_wev[i, :] = wev
    res = np.dot(accum_wev.T, hankel_weights)
    return res.ravel()


def entity_rmatvec(arranged_position_index, idx, factors, entity_key, scaling=None):
    entities = ['users', 'items']
    entity_mode = entities.index(entity_key)
    other_entity = entities[1-entity_mode]
    def rmatvec(vec):
        other_factors = factors[other_entity]
        hankel_weights = factors['hankel_weights']
        if scaling is not None:
            vec = vec * scaling
        result = entity_fiber_rmatvec(arranged_position_index, idx, entity_mode, other_factors, hankel_weights, vec)
        return result
    return rmatvec


def hankel_hooi(
        idx, shape, mlrank, attention_span,
        # attention_matrix = None,
        # scaling_weights = None,
        max_iters = 20,
        growth_tol = 0.001,
        seed = None,
        iter_callback = None,
    ):
    user_rank, item_rank, attn_rank, seqn_rank = mlrank
    n_users, n_items, n_positions = shape
    # assert attention_span == attention_matrix.shape[0]
    sequences_size = n_positions - attention_span + 1

    arranged_indices = arrange_indices(idx, [True, True, True], shape=shape)
    _, arranged_user_index = arranged_indices[0]
    _, arranged_item_index = arranged_indices[1]
    _, arranged_position_index = arranged_indices[2]
    
    factors_store = {}
    rnd = np.random.RandomState(seed)
    factors_store['users'] = np.empty((n_users, user_rank), dtype=np.float64) # only to initialize linear operators
    # factors_store['items'] = scaling_weights[:, np.newaxis] * initialize_columnwise_orthonormal((n_items, item_rank), rnd)
    factors_store['items'] =initialize_columnwise_orthonormal((n_items, item_rank), rnd)
    # attn_factors = factors_store['attention'] = attention_matrix.dot(initialize_columnwise_orthonormal((attention_span, attn_rank), rnd))
    attn_factors = factors_store['attention'] = initialize_columnwise_orthonormal((attention_span, attn_rank), rnd)
    seqn_factors = factors_store['sequences'] = initialize_columnwise_orthonormal((sequences_size, seqn_rank), rnd)

    factors_store['accum_w_i'] = np.empty((n_positions, user_rank*item_rank), dtype=np.float64) # only to initialize linear operators
    factors_store['hankel_weights'] = np.empty((n_positions, attn_rank*seqn_rank), dtype=np.float64) # only to initialize linear operators

    # attn_matvec = series_matvec(factors_store, 'attention', attention_matrix.tocsc())
    # attn_rmatvec = series_rmatvec(factors_store, 'attention', attention_matrix.tocsr())
    attn_matvec = series_matvec(factors_store, 'attention', None)
    attn_rmatvec = series_rmatvec(factors_store, 'attention', None)
    attn_linop = LinearOperator((attention_span, seqn_rank*user_rank*item_rank), attn_matvec, attn_rmatvec)
    
    seqn_matvec = series_matvec(factors_store, 'sequences')
    seqn_rmatvec = series_rmatvec(factors_store, 'sequences')
    seqn_linop = LinearOperator((sequences_size, attn_rank*user_rank*item_rank), seqn_matvec, seqn_rmatvec)
    
    user_matvec = entity_matvec(arranged_user_index, idx, factors_store, 'users')
    user_rmatvec = entity_rmatvec(arranged_position_index, idx, factors_store, 'users')
    user_linop = LinearOperator((n_users, attn_rank*seqn_rank*item_rank), user_matvec, user_rmatvec)
    
    # item_matvec = entity_matvec(arranged_item_index, idx, factors_store, 'items', scaling_weights)
    # item_rmatvec = entity_rmatvec(arranged_position_index, idx, factors_store, 'items', scaling_weights)
    item_matvec = entity_matvec(arranged_item_index, idx, factors_store, 'items', None)
    item_rmatvec = entity_rmatvec(arranged_position_index, idx, factors_store, 'items', None)
    item_linop = LinearOperator((n_items, attn_rank*seqn_rank*user_rank), item_matvec, item_rmatvec)
    
    if iter_callback is None:
        iter_callback = core_growth_callback(growth_tol)

    for step in range(max_iters):
        factors_store['hankel_weights'] = hankel_series_compress_2d(attn_factors, seqn_factors)

        u_user, *_ = svds(user_linop, k=user_rank, return_singular_vectors='u')
        user_factors = factors_store['users'] = np.ascontiguousarray(u_user)
        
        u_item, *_ = svds(item_linop, k=item_rank, return_singular_vectors='u')
        # item_factors = factors_store['items'] = np.ascontiguousarray(u_item * scaling_weights[:, np.newaxis])    
        item_factors = factors_store['items'] = np.ascontiguousarray(u_item)

        factors_store['accum_w_i'] = accum_kron_weights(arranged_position_index, idx, user_factors, item_factors)

        u_attn, *_ = svds(attn_linop, k=attn_rank, return_singular_vectors='u')
        # attn_factors = factors_store['attention'] = np.ascontiguousarray(attention_matrix.dot(u_attn))
        attn_factors = factors_store['attention'] = np.ascontiguousarray(u_attn)
        
        u_seqn, ss, _ = svds(seqn_linop, k=seqn_rank, return_singular_vectors='u')
        seqn_factors = factors_store['sequences'] = np.ascontiguousarray(u_seqn)
        
        raw_factors = (u_user, u_item, u_attn, u_seqn)
        try:
            core_norm = np.linalg.norm(ss)
            iter_callback(step, core_norm, raw_factors)
        except StopIteration:
            break
    return raw_factors


def exp_decay(decay_factor, n):
    return np.e**(-(n-1)*decay_factor)

def lin_decay(decay_factor, n):
    return n**(-decay_factor)

def attention_weights(decay_factor, cutoff, max_elements=None, exponential_decay=False, reverse=False):
    if (decay_factor == 0 or cutoff == 0) and (max_elements is None or max_elements <= 0):
        raise TESSAError('Infinite sequence.')
    decay_function = exp_decay if exponential_decay else lin_decay
    weights = takewhile(lambda x: x>=cutoff, (decay_function(decay_factor, n) for n in count(1, 1)))
    if max_elements is not None:
        weights = islice(weights, max_elements)
    if reverse:
        return list(reversed(list(weights)))
    return list(weights)

def form_attention_matrix(size, decay_factor, cutoff=0, span=0, exponential_decay=False, reverse=False, format='csc', stochastic_axis=None, dtype=None):
    stochastic = stochastic_axis is not None
    span = min(span or np.iinfo('i8').max, size)
    weights = attention_weights(decay_factor, cutoff=cutoff, max_elements=span, exponential_decay=exponential_decay, reverse=reverse)
    diag_values = [np.broadcast_to(w, size) for w in weights]
    matrix = diags(diag_values, offsets=range(0, -len(diag_values), -1), format=format, dtype=dtype)
    if stochastic:
        scalings = matrix.sum(axis=stochastic_axis).A.squeeze()
        if stochastic_axis == 0:
            matrix = matrix.dot(diags(1./scalings))
        else:
            matrix = diags(1./scalings).dot(matrix)
    return matrix.asformat(format)

def generate_banded_form(matrix):
    matrix = matrix.todia()
    bands = matrix.data
    offsets = matrix.offsets
    num_l = (offsets < 0).sum()
    num_u = (offsets > 0).sum()
    return (num_l, num_u), bands[np.argsort(offsets)[::-1], :]