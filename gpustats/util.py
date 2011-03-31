import numpy as np
import pymc.distributions as pymc_dist
import pycuda.driver as drv
import pycuda

_dev_attr = drv.device_attribute

class DeviceInfo(object):

    def __init__(self):
        self._dev = pycuda.autoinit.device
        #self._dev = drv.Device(dev)
        self._attr = self._dev.get_attributes()

        self.max_block_threads = self._attr[_dev_attr.MAX_THREADS_PER_BLOCK]
        self.shared_mem = self._attr[_dev_attr.MAX_SHARED_MEMORY_PER_BLOCK]
        self.warp_size = self._attr[_dev_attr.WARP_SIZE]
        self.max_registers = self._attr[_dev_attr.MAX_REGISTERS_PER_BLOCK]
        self.compute_cap = self._dev.compute_capability()

HALF_WARP = 16

def random_cov(dim):
    return pymc_dist.rinverse_wishart(dim, np.eye(dim))

def unvech(v):
    # quadratic formula, correct fp error
    rows = .5 * (-1 + np.sqrt(1 + 8 * len(v)))
    rows = int(np.round(rows))

    result = np.zeros((rows, rows))
    result[np.triu_indices(rows)] = v
    result = result + result.T

    # divide diagonal elements by 2
    result[np.diag_indices(rows)] /= 2

    return result

def pad_data(data):
    """
    Pad data to avoid bank conflicts on the GPU-- dimension should not be a
    multiple of the half-warp size (16)
    """
    n, k = data.shape

    if not k % HALF_WARP:
        pad_dim = k + 1
    else:
        pad_dim = k

    if k != pad_dim:
        padded_data = np.empty((n, pad_dim), dtype=np.float32)
        padded_data[:, :k] = data

        return padded_data
    else:
        return prep_ndarray(data)

def prep_ndarray(arr):
    # is float32 and contiguous?
    if not arr.dtype == np.float32 or not arr.flags.contiguous:
        arr = np.array(arr, dtype=np.float32)

    return arr


def tune_sfm(n, k, func_regs ,logged=False):
    """
    Outputs the 'opimal' block and grid configuration
    for the sample from measure kernel.
    """
    info = DeviceInfo()
    comp_cap = info.compute_cap
    max_smem = info.shared_mem * 0.9
    max_threads = int(info.max_block_threads * 0.5)
    max_regs = info.max_registers

    # We want smallest dim possible in x dimsension while
    # still reading mem correctly

    if comp_cap[0] == 1:
        xdim = 16
    else:
        xdim = 32
    

    def sfm_config_ok(xdim, ydim, func_regs, max_regs, max_smem, max_threads):
        ok = 4*(xdim*ydim + 2*ydim) < max_smem and func_regs*ydim*xdim < max_regs
        return ok and xdim*ydim <= max_threads

    ydim = 16
    while sfm_config_ok(xdim, ydim, func_regs, max_regs, max_smem, max_threads):
        ydim += 1

    ydim -= 1
    
    nblocks = int(n/ydim) + 1

    return (nblocks,1), (xdim,ydim,1)
    
    

def tune_blocksize(data, params, func_regs):
    """
    For multivariate distributions-- what's the optimal block size given the
    gpu?

    Parameters
    ----------
    data : ndarray
    params : ndarray

    Returns
    -------
    (data_per, params_per) : (int, int)
    """
    # TODO: how to figure out active device in this thread for the multigpu
    # case?
    info = DeviceInfo()

    max_smem = info.shared_mem * 0.9
    max_threads = int(info.max_block_threads * 0.5)
    max_regs = info.max_registers

    params_per = max_threads
    if (len(params) < params_per):
        params_per = _next_pow2(len(params), info.max_block_threads)

    data_per = max_threads / params_per

    def _can_fit(data_per, params_per):
        ok = compute_shmem(data, params, data_per, params_per) <= max_smem
        return ok and func_regs*data_per*params_per <= max_regs

    while True:
        while not _can_fit(data_per, params_per):
            if data_per <= 1:
                break

            if params_per > 1:
                # reduce number of parameters first
                params_per /= 2
            else:
                # can't go any further, have to do less data
                data_per /= 2

        if data_per == 0:
            # we failed somehow. start over
            data_per = 1
            params_per /= 2
            continue
        else:
            break

    while _can_fit(2 * data_per, params_per):
        if 2 * data_per * params_per < max_threads:
            data_per *= 2
        else:
            # hit block size limit
            break

    return data_per, params_per

def get_boxes(n, box_size):
    # how many boxes of size box_size are needed to hold n things
    return int((n + box_size - 1) / box_size)

def compute_shmem(data, params, data_per, params_per):
    result_space = data_per * params_per

    data_dim = 1 if len(data.shape) == 1 else data.shape[1]
    params_dim = len(params) if len(params.shape) == 1 else params.shape[1]

    param_space = params_dim * params_per
    data_space = data_dim * data_per
    return 4 * (result_space + param_space + data_space)

def _next_pow2(k, pow2):
    while k <= pow2 / 2:
        pow2 /= 2
    return pow2

def next_multiple(k, mult):
    if k % mult:
        return k + (mult - k % mult)
    else:
        return k
