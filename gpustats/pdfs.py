from numpy.linalg import cholesky as chol
import numpy as np
from numpy import linalg

from pycuda.gpuarray import GPUArray, to_gpu
from pycuda.gpuarray import empty as gpu_empty
import gpustats.kernels as kernels
from gpustats import codegen
from gpustats.util import transpose as gpu_transpose
reload(codegen)
reload(kernels)
import gpustats.util as util

__all__ = ['mvnpdf_multi']

cu_module = codegen.get_full_cuda_module()


def _multivariate_pdf_call(cu_func, data, packed_params, get, order,
                           datadim=None):
    packed_params = util.prep_ndarray(packed_params)
    func_regs = cu_func.num_regs

    # Prep the data. Skip if gpu data...
    if isinstance(data, GPUArray):
        padded_data = data
        if datadim is None:
            n_data, dim = data.shape
        else:
            n_data, dim = data.shape[0], datadim

    else:
        n_data, dim = data.shape
        padded_data = util.pad_data(data)

    n_params = len(packed_params)
    data_per, params_per = util.tune_blocksize(
        padded_data,
        packed_params,
        func_regs
    )

    shared_mem = util.compute_shared_mem(
        padded_data,
        packed_params,
        data_per,
        params_per
    )
    block_design = (data_per * params_per, 1, 1)
    grid_design = (util.get_boxes(n_data, data_per),
                   util.get_boxes(n_params, params_per))

    # see cufiles/mvcaller.cu
    design = np.array(
        (
            (data_per, params_per) +  # block design
            padded_data.shape +       # data spec
            (dim,) +                  # non-padded number of data columns
            packed_params.shape       # params spec
        ),
        dtype=np.int32
    )

    if n_params == 1:
        gpu_dest = gpu_empty(n_data, dtype=np.float32)
    else:
        gpu_dest = gpu_empty((n_data, n_params), dtype=np.float32, order='F')

    # Upload data if not already uploaded
    if not isinstance(padded_data, GPUArray):
        gpu_padded_data = to_gpu(padded_data)
    else:
        gpu_padded_data = padded_data

    gpu_packed_params = to_gpu(packed_params)

    params = (gpu_dest, gpu_padded_data, gpu_packed_params) + tuple(design)
    kwargs = dict(block=block_design, grid=grid_design, shared=shared_mem)
    cu_func(*params, **kwargs)

    gpu_packed_params.gpudata.free()
    if get:
        if order == 'F':
            return gpu_dest.get()
        else:
            return np.asarray(gpu_dest.get(), dtype=np.float32, order='C')

    else:
        if order == 'F' or n_params == 1:
            return gpu_dest
        else:
            res = gpu_transpose(
                util.gpu_array_reshape(gpu_dest, (n_params, n_data), "C")
            )
            gpu_dest.gpudata.free()
            return res


def mvnpdf_multi(data, means, covs, weights=None, logged=True,
                 get=True, order="F", datadim=None):
    """
    Multivariate normal density with multiple sets of parameters

    Parameters
    ----------
    data : ndarray (n x k)
    covs : sequence of 2d k x k matrices (length j)
    weights : ndarray (length j)
        Multiplier for component j, usually will sum to 1

    get = False leaves the result on the GPU
    without copying back.

    If data has already been padded, the original dimension
    must be passed in datadim

    It data is of GPUarray type, the data is assumed to be
    padded, and datadim will need to be passed if padding
    was needed.

    Returns
    -------
    densities : n x j
    """
    if logged:
        cu_func = cu_module.get_function('log_pdf_mvnormal')
    else:
        cu_func = cu_module.get_function('pdf_mvnormal')

    assert(len(covs) == len(means))

    ichol_sigmas = [linalg.inv(chol(c)) for c in covs]
    logdets = [-2.0*np.log(c.diagonal()).sum() for c in ichol_sigmas]

    if weights is None:
        weights = np.ones(len(means))

    packed_params = _pack_mvnpdf_params(means, ichol_sigmas, logdets, weights)

    return _multivariate_pdf_call(cu_func, data, packed_params,
                                  get, order, datadim)


def _pack_mvnpdf_params(means, ichol_sigmas, logdets, weights):
    to_pack = []
    for m, ch, ld, w in zip(means, ichol_sigmas, logdets, weights):
        to_pack.append(_pack_mvnpdf_params_single(m, ch, ld, w))

    return np.vstack(to_pack)


def _pack_mvnpdf_params_single(mean, ichol_sigma, logdet, weight=1):
    pad_multiple = 16
    k = len(mean)
    mean_len = k
    ichol_len = k * (k + 1) / 2
    mch_len = mean_len + ichol_len

    packed_dim = util.next_multiple(mch_len + 2, pad_multiple)

    packed_params = np.empty(packed_dim, dtype=np.float32)
    packed_params[:mean_len] = mean

    packed_params[mean_len:mch_len] = ichol_sigma[np.tril_indices(k)]
    packed_params[mch_len:mch_len + 2] = weight, logdet

    return packed_params
