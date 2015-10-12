import unittest

from numpy.random import randn
from numpy.testing import assert_almost_equal
import numpy as np

from scipy import linalg
from pymc.distributions import rwishart
from pymc import mv_normal_cov_like as pdf_func

import gpustats as gps

DECIMAL_6 = 6
DECIMAL_5 = 5
DECIMAL_4 = 4
DECIMAL_3 = 3
DECIMAL_2 = 2

np.set_printoptions(suppress=True)


def random_cov(dim):
    return linalg.inv(rwishart(dim, np.eye(dim)))


def python_mvnpdf(data, means, covs):

    results = []
    for i, datum in enumerate(data):
        for j, cov in enumerate(covs):
            mean = means[j]
            results.append(pdf_func(datum, mean, cov))

    return np.array(results).reshape((len(data), len(covs))).squeeze()


def _make_test_case(n=1000, k=4, p=1):
    data = randn(n, k)
    covs = [random_cov(k) for _ in range(p)]
    means = [randn(k) for _ in range(p)]
    return data, means, covs

# debugging...


def _compare_multi(n, k, p):
    data, means, covs = _make_test_case(n, k, p)

    # cpu in PyMC
    py_result = python_mvnpdf(data, means, covs)

    # gpu
    result = gps.mvnpdf_multi(data, means, covs)

    return result, py_result


class TestMVN(unittest.TestCase):
    # n data, dim, n components
    test_cases = [(1000, 4, 1),
                  (1000, 4, 16),
                  (1000, 4, 32),
                  (1000, 4, 64),
                  (1000, 7, 64),
                  (1000, 8, 64),
                  (1000, 14, 32),
                  (1000, 16, 128),
                  (250, 25, 32),
                  (10, 15, 2),
                  (500000, 5, 12)]

    def _check_multi(self, n, k, p):
        a, b = _compare_multi(n, k, p)
        assert_almost_equal(a, b, DECIMAL_2)

    def test_multi(self):
        for n, k, p in self.test_cases:
            self._check_multi(n, k, p)


if __name__ == '__main__':
    _compare_multi(500000, 4, 128)
    pass
