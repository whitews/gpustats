import pycuda.driver as drv
drv.init()
if drv.Context.get_current() is None:
    import pycuda.autoinit

import os
from pycuda.compiler import SourceModule
from gpustats.util import get_cufiles_path


class CUDAModule(object):
    """
    Interfaces with PyCUDA

    Parameters
    ----------
    kernel_dict :
    """
    def __init__(self, kernel_dict):
        self.kernel_dict = kernel_dict
        self.support_code = _get_support_code()

        self.all_code = self._get_full_source()
        try:
            # dict mapping contexts to their respective loaded code modules
            self.pycuda_modules = {
                drv.Context.get_current(): SourceModule(self.all_code)
            }
        except Exception:
            f = open('foo.cu', 'w')
            print >> f, self.all_code
            f.close()
            raise

    def _get_full_source(self):
        formatted_kernels = [kern.get_code()
                             for kern in self.kernel_dict.values()]
        return '\n'.join([self.support_code] + formatted_kernels)

    def get_function(self, name):
        # get the module for this context
        context = drv.Context.get_current()
        try:
            mod = self.pycuda_modules[context]
        except KeyError:
            # if it's a new context, init the module
            self.pycuda_modules[context] = SourceModule(self.all_code)
            mod = self.pycuda_modules[context]
        return mod.get_function('k_%s' % name)


def _get_support_code():
    path = os.path.join(get_cufiles_path(), 'support.cu')
    return open(path).read()


def _get_mvcaller_code():
    # for multivariate PDFs
    path = os.path.join(get_cufiles_path(), 'mvcaller.cu')
    return open(path).read()


class Kernel(object):

    def __init__(self, name):
        if name is None:
            raise ValueError('Kernel must have a default name')

        self.name = name

    def get_code(self):
        logic = self.get_logic()
        caller = self.get_caller()
        return '\n'.join((logic, caller))

    def get_logic(self, **kwargs):
        raise NotImplementedError

    def get_caller(self, **kwargs):
        raise NotImplementedError

    def get_name(self, name=None):
        # can override default name, for transforms. this a hack?
        if name is None:
            name = self.name

        return name


class CUFile(Kernel):
    """
    Expose kernel contained in .cu file in the cufiles directory to code
    generation framework. Kernel need only have a template to be able to change
    the name of the generated kernel
    """
    def __init__(self, name, file_path):
        self.full_path = os.path.join(
            get_cufiles_path(),
            file_path
        )

        Kernel.__init__(self, name)

    def get_code(self):
        code = open(self.full_path).read()
        return code % {'name': self.name}

    def get_logic(self, **kwargs):
        raise NotImplementedError

    def get_caller(self, **kwargs):
        raise NotImplementedError


class MVDensityKernel(Kernel):
    """
    Generate kernel for multi-variate probability density function
    """
    _caller = _get_mvcaller_code()

    def __init__(self, name, logic_code):
        self.logic_code = logic_code
        Kernel.__init__(self, name)

    def get_logic(self, name=None):
        return self.logic_code % {'name': self.get_name(name)}

    def get_caller(self, name=None):
        return self._caller % {'name': self.get_name(name)}


class Exp(Kernel):
    op = 'expf'

    def __init__(self, name, kernel):
        self.kernel = kernel
        Kernel.__init__(self, name)

    def get_logic(self, name=None):
        name = self.get_name(name)

        actual_name = '%s_stub' % name
        kernel_logic = self.kernel.get_logic(name=actual_name)

        stub_caller = """
__device__ float %(name)s(float* x, float* params, int dim) {
    return %(op)s(%(actual_kernel)s(x, params, dim));
}
"""

        transform_logic = stub_caller % {
            'name': name,
            'actual_kernel': actual_name,
            'op': self.op
        }

        return '\n'.join((kernel_logic, transform_logic))

    def get_caller(self):
        return self.kernel.get_caller(self.name)

_cu_module = None


def get_full_cuda_module():
    import gpustats.kernels as kernels
    global _cu_module

    if _cu_module is None:
        all_kernels = dict(
            (k, v)
            for k, v in kernels.__dict__.iteritems()
            if isinstance(v, Kernel)
        )
        _cu_module = CUDAModule(all_kernels)

    return _cu_module
