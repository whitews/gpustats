import numpy as np

from gpustats import codegen
import gpustats.util as util
from pycuda.gpuarray import GPUArray, to_gpu
from pycuda.gpuarray import empty as gpu_empty

cu_module = codegen.get_full_cuda_module()


def sample_discrete(densities, logged=False, return_gpuarray=False):
    """
    Takes a categorical sample from the un-normalized univariate
    densities defined in the rows of 'densities'

    Parameters
    ---------
    densities : ndarray or gpuarray (n, k)
    logged: boolean indicating whether densities is on the
    log scale ...

    Returns
    -------
    indices : ndarray or gpuarray (if return_gpuarray=True)
    of length n and dtype = int32
    """

    from gpustats.util import info

    n, k = densities.shape

    # prep data
    if isinstance(densities, GPUArray):
        if densities.flags.f_contiguous:
            gpu_densities = util.transpose(densities)
        else:
            gpu_densities = densities
    else:
        densities = util.prep_ndarray(densities)
        gpu_densities = to_gpu(densities)

    # get gpu function
    cu_func = cu_module.get_function('sample_discrete')

    # setup GPU data
    gpu_random = to_gpu(np.asarray(np.random.rand(n), dtype=np.float32))
    gpu_dest = gpu_empty(n, dtype=np.int32)
    dims = np.array([n, k, logged], dtype=np.int32)

    if info.max_block_threads < 1024:
        x_block_dim = 16
    else:
        x_block_dim = 32

    y_block_dim = 16

    # setup GPU call
    block_design = (x_block_dim, y_block_dim, 1)
    grid_design = (int(n/y_block_dim) + 1, 1)

    shared_mem = 4 * ((x_block_dim+1) * y_block_dim + 2 * y_block_dim)

    cu_func(gpu_densities, gpu_random, gpu_dest, 
            dims[0], dims[1], dims[2], 
            block=block_design, grid=grid_design, shared=shared_mem)

    gpu_random.gpudata.free()
    if return_gpuarray:
        return gpu_dest
    else:
        res = gpu_dest.get()
        gpu_dest.gpudata.free()
        return res
