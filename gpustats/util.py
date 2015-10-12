import os
import sys
import numpy as np
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.tools import context_dependent_memoize
from pycuda.compiler import SourceModule

drv.init()

if drv.Context.get_current() is None:
    import pycuda.autoinit


def threadSafeInit(device=0):
    """
    If gpustats (or any other pycuda work) is used inside a 
    multiprocessing.Process, this function must be used inside the
    thread to clean up invalid contexts and create a new one on the 
    given device. Assumes one GPU per thread.
    """

    import atexit
    drv.init()  # just in case

    # clean up all contexts. most will be invalid from
    # multiprocessing fork
    clean = False
    while not clean:
        _old_ctx = drv.Context.get_current()
        if _old_ctx is None:
            clean = True
        else:
            # detach: will give warnings to stderr if invalid
            _old_c_err = os.dup(sys.stderr.fileno())
            _nl = os.open(os.devnull, os.O_RDWR)
            os.dup2(_nl, sys.stderr.fileno())
            _old_ctx.detach() 
            sys.stderr = os.fdopen(_old_c_err, "wb")
            os.close(_nl)

    from pycuda.tools import clear_context_caches
    clear_context_caches()
        
    # init a new device
    dev = drv.Device(device)
    ctx = dev.make_context()

    # pycuda.autoinit exitfunc is bad now .. delete it
    exit_funcs = atexit._exithandlers
    for fn in exit_funcs:
        if hasattr(fn[0], 'func_name'):
            if fn[0].func_name == '_finish_up':
                exit_funcs.remove(fn)
            if fn[0].func_name == 'clean_all_contexts':  # avoid duplicates
                exit_funcs.remove(fn)

    # make sure we clean again on exit
    atexit.register(clean_all_contexts)


def clean_all_contexts():

    ctx = True
    while ctx is not None:
        ctx = drv.Context.get_current()
        if ctx is not None:
            ctx.detach()

    from pycuda.tools import clear_context_caches
    clear_context_caches()


def gpu_array_reshape(g_array, shape=None, order="C"):
    if shape is None:
        shape = g_array.shape
    return gpuarray.GPUArray(
        shape=shape,
        dtype=g_array.dtype,
        allocator=g_array.allocator,
        base=g_array,
        gpudata=int(g_array.gpudata),
        order=order)

_dev_attr = drv.device_attribute
# TODO: should be different for each device .. assumes they are the same


class DeviceInfo(object):

    def __init__(self):
        self._dev = drv.Context.get_device()
        self._attr = self._dev.get_attributes()

        self.max_block_threads = self._attr[_dev_attr.MAX_THREADS_PER_BLOCK]
        self.shared_mem = self._attr[_dev_attr.MAX_SHARED_MEMORY_PER_BLOCK]
        self.warp_size = self._attr[_dev_attr.WARP_SIZE]
        self.max_registers = self._attr[_dev_attr.MAX_REGISTERS_PER_BLOCK]
        self.compute_cap = self._dev.compute_capability()
        self.max_grid_dim = (self._attr[_dev_attr.MAX_GRID_DIM_X],
                             self._attr[_dev_attr.MAX_GRID_DIM_Y])

info = DeviceInfo()

HALF_WARP = 16


def pad_data(data):
    """
    Pad data to avoid bank conflicts on the GPU-- dimension should not be a
    multiple of the half-warp size (16)
    """
    if type(data) == gpuarray:
        data = data.get()

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
        arr = np.array(arr, dtype=np.float32, order='C')

    return arr


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

    max_smem = info.shared_mem * 0.9
    max_threads = int(info.max_block_threads * 0.5)
    max_regs = info.max_registers
    max_grid = int(info.max_grid_dim[0])

    params_per = 64  # max_threads
    if len(params) < params_per:
        params_per = _next_pow2(len(params), info.max_block_threads)

    min_data_per = data.shape[0] / max_grid
    data_per0 = _next_pow2(max(max_threads / params_per, min_data_per), 512)
    data_per = data_per0

    def _can_fit(data_per, params_per):
        ok = compute_shmem(data, params, data_per, params_per) <= max_smem
        ok = ok and data_per*params_per <= max_threads
        return ok and func_regs*data_per*params_per <= max_regs

    while True:
        while not _can_fit(data_per, params_per):
            if data_per <= min_data_per:
                break

            if params_per > 1:
                # reduce number of parameters first
                params_per /= 2
            else:
                # can't go any further, have to do less data
                data_per /= 2

        if data_per <= min_data_per:
            # we failed somehow. start over
            data_per = 2 * data_per0
            params_per /= 2
            continue
        else:
            break

    while _can_fit(2 * data_per, params_per):
        data_per *= 2

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


def next_multiple(k, multiple):
    if k % multiple:
        return k + (multiple - k % multiple)
    else:
        return k


def get_cufiles_path():
    import os.path as pth
    base_path = pth.abspath(pth.split(__file__)[0])
    return pth.join(base_path, 'cufiles')


@context_dependent_memoize
def _get_transpose_kernel():

    if info.max_block_threads >= 1024:
        t_block_size = 32
    else:
        t_block_size = 16

    import os.path as pth
    mod = SourceModule( 
        open(
            pth.join(get_cufiles_path(), "transpose.cu")
        ).read() % {"block_size": t_block_size}
    )

    func = mod.get_function("transpose")
    func.prepare("PPii")
    return t_block_size, func


def _transpose(tgt, src):
    block_size, func = _get_transpose_kernel()

    h, w = src.shape
    assert tgt.shape == (w, h)
    
    gw = int(np.ceil(float(w) / block_size))
    gh = int(np.ceil(float(h) / block_size))

    func.prepared_call(
        (gw, gh),
        (block_size, block_size, 1),
        tgt.gpudata,
        src.gpudata,
        w,
        h
    )


def transpose(src):
    h, w = src.shape

    result = gpuarray.empty((w, h), dtype=src.dtype)
    _transpose(result, src)
    del src
    return result
