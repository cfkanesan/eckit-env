# Copyright (c) 2026 MeteoSwiss, contributors listed in AUTHORS
#
# Distributed under the terms of the BSD 3-Clause License.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the grid_interp module"""

import pytest
import numpy as np


def test_interp_temp():
    """Simple testcase for the interp_temp function from grid_interp module"""

    import grid_interp as gri

    # create flat model coords
    x = np.tile(np.arange(0, 100), (100, 1))
    y = x.T
    alt_neighbor = np.arange(25).reshape(5, 5)
    z = np.tile(alt_neighbor, (20, 20))
    z = z + np.floor(np.arange(100) / 5)
    coords_flat = np.c_[x.flatten(), y.flatten(), z.flatten()]

    # create pois coords
    poi_xy = np.arange(0, 100, 5) + 2
    delta_h = np.concatenate((np.linspace(-2, 2, 10), np.arange(-18, -13), np.arange(13, 18)))
    poi_z = 12 + delta_h + np.arange(0, 20)
    poi_coords = np.c_[poi_xy, poi_xy, poi_z]

    # ***case with z_fact=1***

    # create flat model data
    data = 4 + 3 * z
    data[:50, :] += 2 * z[:50, :] ** 2 + z[:50, :] ** 3
    data_flat = data.flatten()
    data_flat = data_flat[:, None, None]

    # create expected results
    expected = 4 + 3 * poi_z
    expected[:10] += 2 * poi_z[:10] ** 2 + poi_z[:10] ** 3
    expected = expected[:, None, None]

    # call tested function
    results = gri.interp_temp(coords_flat, data_flat, poi_coords, z_fact=1, k=9)

    # compare
    assert np.allclose(expected, results)

    # ***case with pois altitude all >500m above highest grid point***

    # create expected results
    expected = gri.interp_temp_fixgrad(coords_flat, data_flat, poi_coords + 600, r_search=2, z_weight=1)

    # call tested function
    results = gri.interp_temp(coords_flat, data_flat, poi_coords + 600, z_fact=1, k=25)

    # compare
    assert np.allclose(expected, results)

    # ***case with z_fact>>1***

    # create flat model data
    b = 10
    a = 10
    c = 1000
    d = 2000
    data = np.ones(x.shape)
    data[:, -1] = data[:, -1] * np.arange(x.shape[1]) * b + c
    data_flat = data.flatten()
    data_flat = data_flat[:, None, None]
    poi_coords_new = poi_coords.copy()
    poi_coords_new[:, -1] = 2981
    z_new = np.zeros(z.shape)
    z_new[:, -1] = np.arange(z.shape[1]) * a + d
    coords_flat_new = np.c_[(x * 10).flatten(), (y * 10).flatten(), z_new.flatten()]

    expected = c - d + b / a * 2981

    # call tested function
    results = gri.interp_temp(coords_flat_new, data_flat, poi_coords_new, z_fact=50, k=25, grad_only=False)

    # compare
    assert np.allclose(expected, results)

    # ***case with grad_only = True and z_fact=1***

    # create flat model data
    data = 1 + 0.008 * z
    data[:50, :] = 8 + 0.008 * z[:50, :]
    data_flat = data.flatten()
    data_flat = data_flat[:, None, None]

    # create expected results
    expected = data[poi_xy, poi_xy] + 0.008 * (poi_z - z[poi_xy, poi_xy])
    expected = expected[:, None, None]

    # call tested function
    results = gri.interp_temp(coords_flat, data_flat, poi_coords, z_fact=1, k=15, grad_only=True)

    # compare
    assert np.allclose(expected, results)

    # ***case with grad_only = True and z_fact>>1***

    # create flat model data
    b = 0.2
    a = 10
    data = 10.0 + x
    data[-1, :] = data[-1, :] * b
    data_flat = data.flatten()
    data_flat = data_flat[:, None, None]
    poi_coords[:, -1] = 506
    z_new = np.zeros(z.shape)
    z_new[-1, :] = np.arange(z.shape[1]) * a + 500
    coords_flat = np.c_[(x * 10).flatten(), (y * 10).flatten(), z_new.flatten()]

    expected = data[-1, :][abs(z_new[-1, :] - 506).argmin()] - b / a * abs(z_new[-1, :] - 506).min()

    # call tested function
    results = gri.interp_temp(coords_flat, data_flat, poi_coords, z_fact=500, k=10, grad_only=True)

    # compare
    assert np.allclose(expected, results)

    # ***case with grad_only = True and grad is clipped to 0.03***

    # create flat model data ()
    b = 5
    a = 10
    data = 10.0 + x
    data[-1, :] = data[-1, :] * b
    data_flat = data.flatten()
    data_flat = data_flat[:, None, None]
    poi_coords[:, -1] = 506
    z_new = np.zeros(z.shape)
    z_new[-1, :] = np.arange(z.shape[1]) * a + 500
    coords_flat = np.c_[(x * 10).flatten(), (y * 10).flatten(), z_new.flatten()]

    expected = (
        data[-1, :][abs(z_new[-1, :] - 506).argmin()] - np.clip(b / a, -0.01, 0.03) * abs(z_new[-1, :] - 506).min()
    )

    # call tested function
    results = gri.interp_temp(coords_flat, data_flat, poi_coords, z_fact=500, k=10, grad_only=True)

    # compare
    assert np.allclose(expected, results)

    # ***flat orography: all neighbor altitudes identical → deg=1 rank-deficient***
    # must raise ValueError without infinite recursion

    x_flat = np.tile(np.arange(0, 10), (10, 1))
    y_flat = x_flat.T
    z_flat = np.zeros(x_flat.shape)  # completely flat
    coords_flat_flat = np.c_[x_flat.flatten(), y_flat.flatten(), z_flat.flatten()]
    data_flat_flat = (x_flat + y_flat).flatten().astype(np.float32)[:, None]
    poi_coords_flat = np.array([[2.5, 2.5, 0.0], [7.0, 7.0, 0.0]])

    with pytest.raises(ValueError, match="rank-deficient"):
        gri.interp_temp(coords_flat_flat, data_flat_flat, poi_coords_flat, z_fact=1, k=9)

    # ***chunk_size invariance: results must not depend on chunk_size***

    # use the last coords_flat / data_flat / poi_coords from the previous case
    results_default = gri.interp_temp(coords_flat, data_flat, poi_coords, z_fact=1, k=9)
    for cs in (1, 3, len(poi_coords)):
        results_cs = gri.interp_temp(coords_flat, data_flat, poi_coords, z_fact=1, k=9, chunk_size=cs)
        assert np.allclose(results_default, results_cs), f"Results differ with chunk_size={cs}"


def test_sounding_batch():
    """
    Simple testcase for the _sounding_batch function from grid_interp module
    """

    import grid_interp as gri

    # create test data — wrap single-POI inputs in a batch dimension of size 1
    x = np.arange(-5, 10, 0.1)
    func1 = lambda x: 2 * np.power(x, 3) + 3 * x - 2
    func2 = lambda x: 2 * x + 4
    y1 = np.tile(func1(x), (4, 6, 10, 1)).T
    y2 = np.tile(func2(x), (4, 6, 10, 1)).T
    y = np.stack((y1, y2), axis=-1)
    x_pred = np.array([3.0])

    # expected results
    y_exp1 = np.ones((10, 6, 4)) * func1(x_pred)
    y_exp2 = np.ones((10, 6, 4)) * func2(x_pred)
    y_exp = np.stack((y_exp1, y_exp2), axis=-1)  # shape (10, 6, 4, 2)

    # _sounding_batch expects (n_sub, k, ...) — wrap in batch dim of 1
    y_batch = y[None]  # (1, k, 10, 6, 4, 2)
    x_batch = x[None]  # (1, k)
    poi_alts = x_pred  # (1,)

    # call tested function
    y_pred = gri._sounding_batch(y_batch, x_batch, poi_alts, deg=3, w=None)

    # compare — output shape is (1, 10, 6, 4, 2)
    assert np.allclose(y_exp, y_pred[0])

    # test rank-deficient fallback: deg=3 with repeated x values falls back to deg=1
    y_rd = np.c_[[3, 3, 3, 3, -1], [2, 2, 2, 2, 2]]  # (5, 2)
    x_rd = np.array([0.0, 1.0, 2.0, 2.0, 2.0])
    y_exp_rd = np.array([1.375, 2.0])

    y_rd_batch = y_rd[None]  # (1, 5, 2)
    x_rd_batch = x_rd[None]  # (1, 5)
    poi_alts_rd = np.array([2.5])

    y_pred3 = gri._sounding_batch(y_rd_batch, x_rd_batch, poi_alts_rd, deg=3)
    y_pred1 = gri._sounding_batch(y_rd_batch, x_rd_batch, poi_alts_rd, deg=1)
    np.testing.assert_allclose(y_pred3[0], y_pred1[0], rtol=1e-5)
    np.testing.assert_allclose(y_exp_rd, y_pred1[0], rtol=1e-5)


def test_interp_temp_fixgrad():
    """
    Simple testcase for the interp_temp_fixgrad function from grid_interp
    module
    """

    import grid_interp as gri

    grad = 0.006

    # create flat model coords
    x = np.tile(np.arange(10), (10, 1))
    y = x.T
    z = x + 1000
    coords_flat = np.c_[x.flatten(), y.flatten(), z.flatten()]

    # create pois coords
    poi_x = np.array([0, 5, 9])
    poi_z = np.array([900, 1010, 1100])
    poi_coords = np.c_[poi_x, poi_x, poi_z]

    # create flat model data
    data_flat = np.zeros(coords_flat.shape)

    # create expected results
    expected = (z[poi_x, poi_x] - poi_z) * grad
    expected = expected[:, None]

    # call tested function
    results = gri.interp_temp_fixgrad(coords_flat, data_flat, poi_coords, r_search=1, grad=grad)

    # compare
    assert np.allclose(expected, results)


def test_grid_assignation():
    """
    Simple testcase for the _grid_assignation function from grid_interp module
    """

    import grid_interp as gri

    # create flat model coords
    x = np.arange(10)
    coords_flat = np.c_[x, x, x]

    # create pois coords
    poi_x = x[4:6]
    poi_coords = np.c_[poi_x, poi_x, poi_x + 2]

    # call tested function
    results = gri._grid_assignation(coords_flat, poi_coords, r_search=1, z_weight=500, deltah_min=0)

    # compare
    assert np.allclose(poi_x + 1, results)

    # call tested function
    results = gri._grid_assignation(coords_flat, poi_coords, r_search=1.5, z_weight=500, deltah_min=0)

    # compare
    assert np.allclose(poi_x + 2, results)

    # call tested function
    results = gri._grid_assignation(coords_flat, poi_coords, r_search=1.5, z_weight=1, deltah_min=0)

    # compare
    assert np.allclose(poi_x, results)

    # call tested function
    results = gri._grid_assignation(coords_flat, poi_coords, r_search=1.5, z_weight=500, deltah_min=1)

    # compare
    assert np.allclose(poi_x + 1, results)


def test_compute_dist_weight():
    """
    Simple testcase for the _compute_dist_weight function from grid_interp
    module
    """

    import grid_interp as gri

    dist = np.random.rand(10, 5)
    weights = gri._compute_dist_weight(dist, 2, np.float32)
    assert np.allclose(np.sum(weights, axis=1), 1.0)
    weights2 = gri._compute_dist_weight(dist, 0, np.float64)
    assert np.all(weights2 == 1 / 5)


def test_nn_lookup():
    """
    Simple testcase for the _compute_dist_weight function from grid_interp
    module
    """

    import grid_interp as gri

    k = 4

    # create flat model coords
    x = np.tile(np.arange(0, 10), (10, 1))
    y = x.T
    coords_flat = np.c_[x.flatten(), y.flatten()]

    # create pois coords
    poi_x = np.array([0.1, 5.6])
    poi_y = np.array([0.3, 8.7])
    poi_coords = np.c_[poi_x, poi_y]

    # create expected results
    exp_ii_1 = np.ravel_multi_index((np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])), x.shape)
    exp_ii_2 = np.ravel_multi_index((np.array([9, 9, 8, 8]), np.array([6, 5, 6, 5])), x.shape)
    expected_ii = np.vstack((exp_ii_1, exp_ii_2))
    expected_dd = np.array(
        [
            [np.linalg.norm(point - coords_flat[ind]) for ind in expected_ii[ind_poi]]
            for ind_poi, point in enumerate(poi_coords)
        ]
    )

    # call tested function
    results_dd, results_ii = gri.nn_lookup(coords_flat, poi_coords, k)

    # compare
    assert np.allclose(expected_dd, results_dd)
    assert np.allclose(expected_ii, results_ii)
