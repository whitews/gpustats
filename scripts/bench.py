from pandas import *

import numpy as np
import timeit

data = np.random.randn(1000000)
mean = 20
std = 5

univ_setup = """
import numpy as np
from pycuda.gpuarray import to_gpu
k = 8
means = np.random.randn(k)
stds = np.abs(np.random.randn(k))

mean = 20
std = 5
import gpustats
from scipy.stats import norm
cpu_data = np.random.randn(%d)
gpu_data = cpu_data
"""

univ_setup_gpuarray = univ_setup + """
gpu_data = to_gpu(cpu_data)
"""


def get_timeit(stmt, setup, iter=10):
    return timeit.Timer(stmt, setup).timeit(number=iter) / iter


def compare_timings_single(n, setup=univ_setup):
    gpu = "gpustats.normpdf(gpu_data, mean, std, logged=False)"
    cpu = "norm.pdf(cpu_data, loc=mean, scale=std)"
    setup = setup % n
    return {'GPU': get_timeit(gpu, setup, iter=1000),
            'CPU': get_timeit(cpu, setup)}


def compare_timings_multi(n, setup=univ_setup):
    gpu = "gpustats.normpdf_multi(gpu_data, means, stds, logged=False)"
    cpu = """
for m, s in zip(means, stds):
    norm.pdf(cpu_data, loc=m, scale=s)
"""
    setup = setup % n
    return {'GPU': get_timeit(gpu, setup, iter=100),
            'CPU': get_timeit(cpu, setup)}


def get_timing_results(timing_f):
    lengths = [100, 1000, 10000, 100000, 1000000]

    result = {}
    for n in lengths:
        print n
        result[n] = timing_f(n)
    result = DataFrame(result).T
    result['Speedup'] = result['CPU'] / result['GPU']
    return result

single = get_timing_results(compare_timings_single)
comp_gpu = lambda n: compare_timings_single(n, setup=univ_setup_gpuarray)
single_gpu = get_timing_results(comp_gpu)
multi = get_timing_results(compare_timings_multi)
comp_gpu = lambda n: compare_timings_multi(n, setup=univ_setup_gpuarray)
multi_gpu = get_timing_results(comp_gpu)

data = DataFrame({
    'Single': single['Speedup'],
    'Single (GPUArray)': single_gpu['Speedup'],
    'Multi': multi['Speedup'],
    'Multi (GPUArray)': multi_gpu['Speedup'],
})
