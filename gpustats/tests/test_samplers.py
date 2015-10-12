import unittest

from numpy.random import rand
from numpy.testing import assert_almost_equal
import numpy as np

import gpustats.sampler as gpu_sampler


DECIMAL_6 = 6
DECIMAL_5 = 5
DECIMAL_4 = 4
DECIMAL_3 = 3
DECIMAL_2 = 2
DECIMAL_1 = 1

np.set_printoptions(suppress=True)


def _make_test_densities(n=10000, k=4):
    dens = rand(k)
    densities = [dens.copy() for _ in range(n)]
    return np.asarray(densities)


def _compare_discrete(n, k):
    densities = _make_test_densities(n, k)
    dens = densities[0, :].copy() / densities[0, :].sum()
    expected_mu = np.dot(np.arange(k), dens)

    labels = gpu_sampler.sample_discrete(densities, logged=False)
    est_mu = labels.mean()
    return est_mu, expected_mu


def _compare_logged(n, k):
    densities = np.log(_make_test_densities(n, k))
    dens = np.exp((densities[0, :] - densities[0, :].max()))
    dens = dens / dens.sum()
    expected_mu = np.dot(np.arange(k), dens)

    labels = gpu_sampler.sample_discrete(densities, logged=True)
    est_mu = labels.mean()
    return est_mu, expected_mu


class TestDiscreteSampler(unittest.TestCase):
    test_cases = [(100000, 4),
                  (100000, 9),
                  (100000, 16),
                  (100000, 20),
                  (1000000, 35)]

    def _check_discrete(self, n, k):
        a, b = _compare_discrete(n, k)
        assert_almost_equal(a, b, DECIMAL_1)

    def _check_logged(self, n, k):
        a, b = _compare_logged(n, k)
        assert_almost_equal(a, b, DECIMAL_1)

    def test_discrete(self):
        for n, k in self.test_cases:
            self._check_discrete(n, k)

    def test_logged(self):
        for n, k in self.test_cases:
            self._check_logged(n, k)


if __name__ == '__main__':
    print 'starting sampler'
    a, b = _compare_logged(1000000, 35)
    print a
    print b
