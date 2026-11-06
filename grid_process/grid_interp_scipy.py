"""
Module for spatial interpolation of gridded datasets
"""

from typing import Union, Optional

import numpy as np
from scipy.spatial import KDTree


def interp_temp(
    coords_flat: np.ndarray,
    data_flat: np.ndarray,
    pois_coords: np.ndarray,
    z_fact: int = 25,
    k: int = 36,
    weight_power: Optional[float] = None,
    grad_only: bool = False,
    chunk_size: int = 20,
) -> np.ndarray:
    """
    Interpolation from grid to POIs for temperature data, based on local
    soundings. The grid does not need to be regular. The sounding curves are
    fitted on the k nearest neighbors considering a distance that more heavily
    penalizes elevation differences for ``z_fact`` > 1.

    Parameters
    ----------
    coords_flat : numpy.ndarray, shape (n_grids, m)
        array of flattened grid coordinates. n_grids is the number of model
        grid points, m the number of coordinates dimensions (i.e. 3)
    data_flat : numpy.ndarray, shape (n_grids, ...)
        array containing the flattened model data with various trailing
        dimensions
    pois_coords : numpy.ndarray , shape (n_pois, m)
        array of POI coordinates. n_pois is the number of POIs, m the number of
        coordinates dimensions (i.e. 3)
    z_fact : positive int, optional
        factor to multiply the z coordinates to find nearest neighbors based on a
        non-euclidean distance that penalizes more altitude differences. Default
        value is 50. A value of 1 will use regular euclidean distances.
    k : positive int, optional
        number of considered nearest neighbors for the local sounding. Default
        is 36
    weight_power : positive float, optional
        a weight vector is passed to the sounding fit according to
        ``w = d^(-weight_power)``, with d being the (non-euclidean) distances between
        the considered POI and the k nearest grid points. The greater is the
        ``weight_power``, the stronger is the penalization with distance. A
        ``weight_power`` of 0 is equivalent to None (i.e. same weights for all
        points). By default, gives same weight for all points
    grad_only : bool, optionaldf 
        if True, only uses the local gradient to correct the nearest gridpoint
        temperature based on the altitude difference with the POI. Default is
        False
    chunk_size : positive int, optional
        number of POIs processed per sounding batch. Smaller values reduce peak
        memory (dominant cost: ``chunk_size × k × n_trail × 12`` bytes) and
        keep the working set in L3 cache, which is typically faster when
        ``n_trail`` is large. Larger values reduce loop overhead and are better
        for small trailing dimensions or when n_pois is small. Default is 20.

    Returns
    -------
    poi_data : numpy.ndarray, shape (n_pois, ...)
        array containing the interpolated values and the same trailing
        dimensions as the input ``data_flat`` array

    Raises
    ------
    ValueError : Raised if the orography is flat, as no sounding can be estimated.

    Note
    ----
    For POIs with altitude higher than 300m above the highest gridpoint or lower
    than 500m below the lowest gridpoint in the local neighborhood used for the fit,
    the fixgrad method is used with default gradient of 0.006 [K / m].
    """

    # make sure k is smaller or equal to the number of grid points
    k = min(k, len(coords_flat))

    # rescale z coords
    coords_flat_zfact = coords_flat.copy()
    pois_coords_zfact = pois_coords.copy()
    coords_flat_zfact[:, -1] *= z_fact
    pois_coords_zfact[:, -1] *= z_fact

    # find indexes and distances of k nearest neighbors in the non-euclidean dist
    dist_nn, ind_nn = nn_lookup_cached(coords_flat_zfact, pois_coords_zfact, k)

    # preallocate output array
    data_dtype = data_flat.dtype
    poi_data = np.empty((pois_coords.shape[0],) + data_flat.shape[1:], dtype=data_dtype)

    # gather neighbor altitudes for all POIs at once (small: n_pois × k floats)
    all_alt = coords_flat[ind_nn, -1]  # (n_pois, k)

    if weight_power is not None:
        # weights for the sounding fit according to the distance
        dist_weight: np.ndarray | None = _compute_dist_weight(dist_nn, weight_power, data_dtype)
    else:
        dist_weight = None

    poi_alts = pois_coords[:, -1]

    chunk_size = max(1, chunk_size)

    if grad_only:
        # grad_only always uses a deg=1 fit regardless of altitude position, so
        # the deg3/deg1 split is unnecessary. Only min/max quantiles are needed
        # to classify fixgrad vs sounding POIs.
        alt_minmax = np.quantile(all_alt, [0, 1], axis=1)  # shape (2, n_pois)
        fixgrad_mask = (poi_alts < alt_minmax[0] - 600) | (poi_alts > alt_minmax[1] + 200)
        sounding_groups = [(np.nonzero(~fixgrad_mask)[0], 1, True)]
    else:
        # classify each POI into three groups: fixgrad, deg1 sounding, deg3 sounding
        alt_q = np.quantile(all_alt, [0, 0.15, 0.85, 1], axis=1)  # shape (4, n_pois)
        fixgrad_mask = (poi_alts < alt_q[0] - 600) | (poi_alts > alt_q[-1] + 200)
        sounding_mask = ~fixgrad_mask
        deg3_mask = sounding_mask & (alt_q[1] < poi_alts) & (poi_alts < alt_q[2])
        deg1_mask = sounding_mask & ~deg3_mask
        sounding_groups = [
            (np.nonzero(deg1_mask)[0], 1, False),
            (np.nonzero(deg3_mask)[0], 3, False),
        ]

    # fixgrad group: only needs the nearest neighbor per POI (same for both branches)
    fg_idx = np.nonzero(fixgrad_mask)[0]
    if fg_idx.size > 0:
        poi_data[fg_idx] = _fixgrad(
            data_flat[ind_nn[fg_idx, 0]],
            all_alt[fg_idx, 0],
            poi_alts[fg_idx],
        )

    # sounding groups: process in memory-bounded chunks
    for idx, deg, go in sounding_groups:
        for start in range(0, idx.size, chunk_size):
            chunk_idx = idx[start : start + chunk_size]
            chunk_w = dist_weight[chunk_idx] if dist_weight is not None else None
            poi_data[chunk_idx] = _sounding_batch(
                data_flat[ind_nn[chunk_idx]],
                all_alt[chunk_idx],
                poi_alts[chunk_idx],
                deg=deg,
                w=chunk_w,
                grad_only=go,
            )

    return poi_data


def _normalize_altitudes(
    altitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize altitudes per POI to improve numerical conditioning of the
    polynomial fit.

    Parameters
    ----------
    altitudes : numpy.ndarray, shape (n_sub, k)
        neighbor altitudes for each POI

    Returns
    -------
    alt_norm : numpy.ndarray, shape (n_sub, k)
        normalized altitudes with zero mean and unit standard deviation per POI
    alt_mean : numpy.ndarray, shape (n_sub, 1)
        per-POI altitude mean used for normalization
    alt_std : numpy.ndarray, shape (n_sub, 1)
        per-POI altitude standard deviation used for normalization
    """

    alt_mean = altitudes.mean(axis=1, keepdims=True)
    alt_std = altitudes.std(axis=1, keepdims=True).clip(1e-10)
    return (altitudes - alt_mean) / alt_std, alt_mean, alt_std


def _build_vandermonde(alt_norm: np.ndarray, deg: int) -> np.ndarray:
    """
    Build a batched Vandermonde matrix from normalized altitudes.

    Parameters
    ----------
    alt_norm : numpy.ndarray, shape (n_sub, k)
        normalized neighbor altitudes
    deg : positive int
        polynomial degree

    Returns
    -------
    V : numpy.ndarray, shape (n_sub, k, deg+1)
        Vandermonde matrix with columns [1, x, x^2, ..., x^deg]
    """

    exponents = np.arange(deg + 1, dtype=np.float64)
    return alt_norm[:, :, None] ** exponents


def _apply_weights(
    vandermonde: np.ndarray, values: np.ndarray, w: Union[np.ndarray, None]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply per-POI weights to the Vandermonde matrix and right-hand side.

    Parameters
    ----------
    vandermonde : numpy.ndarray, shape (n_sub, k, deg+1)
        Vandermonde matrix
    values : numpy.ndarray, shape (n_sub, k, n_trail)
        flattened data values
    w : numpy.ndarray, shape (n_sub, k) or None
        weight vector per POI; if None, uniform weights are used

    Returns
    -------
    vw : numpy.ndarray, shape (n_sub, k, deg+1)
        weight-scaled Vandermonde matrix
    yw : numpy.ndarray, shape (n_sub, k, n_trail)
        weight-scaled right-hand side
    """

    if w is not None:
        ww = w[:, :, None].astype(np.float64, copy=False)
        return vandermonde * ww, values * ww
    return vandermonde, values


def _qr_polyfit_batch(
    vw: np.ndarray, yw: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve a batch of weighted least-squares problems via QR decomposition and
    detect rank-deficient systems.

    Parameters
    ----------
    vw : numpy.ndarray, shape (n_sub, k, deg+1)
        weighted Vandermonde matrix
    yw : numpy.ndarray, shape (n_sub, k, n_trail)
        weighted right-hand side

    Returns
    -------
    coeffs : numpy.ndarray, shape (n_sub, deg+1, n_trail)
        polynomial coefficients; zero for rank-deficient systems
    rank_deficient : numpy.ndarray, shape (n_sub,), dtype bool
        True for systems where the R diagonal indicates near-singularity
    """

    q_mat, r_mat = np.linalg.qr(vw)
    qt_b = q_mat.swapaxes(-2, -1) @ yw  # (n_sub, deg+1, n_trail)
    diag_abs = np.abs(np.diagonal(r_mat, axis1=-2, axis2=-1))  # (n_sub, deg+1)
    rank_ok = diag_abs.min(axis=-1) >= 1e-10 * diag_abs.max(axis=-1)
    coeffs = np.zeros_like(qt_b)
    ok_idx = np.nonzero(rank_ok)[0]
    if ok_idx.size > 0:
        coeffs[ok_idx] = np.linalg.solve(r_mat[ok_idx], qt_b[ok_idx])
    return coeffs, ~rank_ok


def _eval_poly_batch(
    coeffs: np.ndarray, poi_norm: np.ndarray, deg: int
) -> np.ndarray:
    """
    Evaluate a batch of polynomials at normalized POI altitudes.

    Parameters
    ----------
    coeffs : numpy.ndarray, shape (n_sub, deg+1, n_trail)
        polynomial coefficients in the normalized altitude space
    poi_norm : numpy.ndarray, shape (n_sub,)
        normalized target altitudes for each POI
    deg : positive int
        polynomial degree

    Returns
    -------
    result : numpy.ndarray, shape (n_sub, n_trail)
        evaluated polynomial values
    """

    exponents = np.arange(deg + 1, dtype=np.float64)
    vander_poi = poi_norm[:, None] ** exponents  # (n_sub, deg+1)
    return (vander_poi[:, None, :] @ coeffs)[:, 0, :]  # (n_sub, n_trail)


def _sounding_batch(
    data_array: np.ndarray,
    altitudes: np.ndarray,
    poi_alts: np.ndarray,
    deg: int = 3,
    w: Union[np.ndarray, None] = None,
    grad_only: bool = False,
) -> np.ndarray:
    """
    Returns predicted temperature values for a batch of POIs according to the
    local sounding, using a batched QR decomposition.

    Each POI has its own neighborhood and independent polynomial fit. All fits
    are computed simultaneously via ``numpy.linalg.qr`` and
    ``numpy.linalg.solve`` to avoid per-POI Python-level dispatch.

    Parameters
    ----------
    data_array : numpy.ndarray, shape (n_sub, k, ...)
        contains the temperature values of the k nearest neighbor grid points
        for each of the n_sub POIs in this batch
    altitudes : numpy.ndarray, shape (n_sub, k)
        contains the altitudes of the k nearest neighbor grid points for each POI
    poi_alts : numpy.ndarray, shape (n_sub,)
        altitude of each POI in this batch
    deg : positive int
        degree of the polynomial fit. Only used if ``grad_only=False``
    w : numpy.ndarray, shape (n_sub, k) or None
        per-POI weight vectors for the polynomial fit. If None, uniform weights
        are used for all points
    grad_only : bool
        if True, only uses the local gradient to correct the temperature of the
        nearest gridpoint based on the altitude difference with the POI.
        In this case, the gradient is bound to [-0.01, 0.03] K m-1. Default is False

    Returns
    -------
    fitted_val : numpy.ndarray, shape (n_sub, ...)
        array of fitted temperatures for each POI in the batch

    See also
    --------
    :func:`interp_temp`
    """

    n_sub = data_array.shape[0]
    trail_shape = data_array.shape[2:]
    n_trail = int(np.prod(trail_shape)) if trail_shape else 1

    # flatten trailing dims and cast to float64 for numerical stability
    values = data_array.reshape(n_sub, data_array.shape[1], n_trail, order="C").astype(
        np.float64, copy=False
    )
    alt_norm, alt_mean, alt_std = _normalize_altitudes(altitudes)

    if grad_only:
        vandermonde = _build_vandermonde(alt_norm, 1)
        vw, yw = _apply_weights(vandermonde, values, w)
        coeffs, _ = _qr_polyfit_batch(vw, yw)
        # convert slope from normalized space back to original altitude units
        slope = (coeffs[:, 1] / alt_std[:, 0, None]).clip(-0.01, 0.03)
        fitted_val = values[:, 0] + slope * (poi_alts[:, None] - altitudes[:, 0, None])
    else:
        vandermonde = _build_vandermonde(alt_norm, deg)
        vw, yw = _apply_weights(vandermonde, values, w)
        coeffs, rank_deficient = _qr_polyfit_batch(vw, yw)

        poi_norm = (poi_alts - alt_mean[:, 0]) / alt_std[:, 0]
        fitted_val = _eval_poly_batch(coeffs, poi_norm, deg).reshape(n_sub, *trail_shape)

        # overwrite rank-deficient results with a deg=1 fallback. If deg is already 1, raise an error instead.
        bad_idx = np.nonzero(rank_deficient)[0]
        if bad_idx.size > 0:
            if deg == 1:
                raise ValueError(
                    f"deg=1 sounding is rank-deficient for {bad_idx.size} POI(s): "
                    "all k neighbor altitudes are identical. Check the surface altitude coordinate."
                )
            fitted_val[bad_idx] = _sounding_batch(
                data_array[bad_idx],
                altitudes[bad_idx],
                poi_alts[bad_idx],
                deg=1,
                w=w[bad_idx] if w is not None else None,
                grad_only=False,
            )

    return fitted_val.reshape((n_sub,) + trail_shape)


def interp_temp_fixgrad(
    coords_flat: np.ndarray,
    data_flat: np.ndarray,
    pois_coords: np.ndarray,
    r_search: float = 2,
    z_weight: int = 200,
    grad: float = 0.006,
) -> np.ndarray:
    """
    Interpolation from grid to POIs for temperature data by taking the
    surrounding gridpoint with the most similar altitude and further correct
    for altitude difference with a fixed gradient.

    Parameters
    ----------
    coords_flat : numpy.ndarray, shape (n_grids, m)
        array of flattened grid coordinates. n_grids is the number of model
        grid points, m the number of coordinates dimensions (i.e. 3)
    data_flat : numpy.ndarray, shape (n_grids, ...)
        array containing the flattened model data with various trailing
        dimensions
    pois_coords : numpy.ndarray , shape (n_pois, m)
        array of POI coordinates. n_pois is the number of POIs, m the number of
        coordinates dimensions (i.e. 3)
    r_search : positive float, optional
        radius in model gridpoints that define the number of neighbors included
        in the search. Default is 4
    z_weight : positive int, optional
        weight for the z axis in the distance computation of the "most adequat"
        gridpoint. ``d = horizontal_dist + z_weight * vertical_dist``.
        Default value is 200
    grad : positive float
        gradient in [K / m] to correct for altitude difference between POI and
        nearest grid point. Default is 0.006 [K / m]

    Returns
    -------
    poi_data : numpy.ndarray, shape (n_pois, ...)
        array containing the interpolated values and the same trailing
        dimensions as the input ``data_flat`` array.
    """

    ind = _grid_assignation(coords_flat, pois_coords, r_search, z_weight)

    return _fixgrad(
        data_flat[ind,],
        coords_flat[ind, -1],
        pois_coords[:, -1],
        grad,
    )


def _fixgrad(
    best_gp_data: np.ndarray,
    best_gp_alt: np.ndarray,
    pois_alt: np.ndarray,
    grad: float = 0.006,
) -> np.ndarray:
    """
    Correct temperature based on altitudinal differences and a fix gradient.

    Parameters
    ----------
    best_gp_data : numpy.ndarray, shape (n_pois, ...)
        contains the temperature values of the most appropriate gridpoint for
        each POI
    best_gp_alt : numpy.ndarray, shape (n_pois,)
        contains the altitude of the most appropriate gridpoint for each POI.
    pois_alt : numpy.ndarray, shape (n_pois,)
        contains the altitude of each POI
    grad : positive float
        gradient in [K / m] to correct for altitude difference between POI and
        nearest grid point. Default is 0.006 [K / m]

    Returns
    -------
    poi_data : numpy.ndarray, shape (n_pois, ...)
        array of altitude corrected temperatures

    See also
    --------
    :func:`interp_temp_fixgrad`
    """

    ndim_trail = best_gp_data.ndim - 1
    delta_t = (best_gp_alt - pois_alt) * grad
    delta_t = delta_t.astype(best_gp_data.dtype, copy=False)

    return best_gp_data + delta_t.reshape(delta_t.shape + (1,) * ndim_trail)


def _grid_assignation(
    coords_flat: np.ndarray,
    pois_coords: np.ndarray,
    r_search: float,
    z_weight: int,
    deltah_min: int = 50,
) -> np.ndarray:
    """
    Returns the index of the "most appropriate" gridpoint based on a distance
    metric that gives a much greater weight to vertical distances. This is the
    standard used for verification in COSMO.
    """

    k = round(np.pi * r_search**2)
    # find indexes and distances of nearest neighbor grids for all POIs
    dist_nn_horz, ind_nn = nn_lookup_cached(coords_flat[:, :2], pois_coords[:, :2], k)

    dist_nn_vert = np.abs(coords_flat[ind_nn, -1] - pois_coords[:, [-1]])
    dist_weighted = dist_nn_horz + z_weight * (dist_nn_vert - deltah_min).clip(0)

    return ind_nn[np.arange(ind_nn.shape[0]), np.argmin(dist_weighted, axis=1)]


def nn_lookup(
    grid_coords: np.ndarray, points: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the k nearest grid points and corresponding distances for a set of
    POIs.

    Parameters
    ----------
    grid_coords : numpy.ndarray, shape (n, m)
        array of flattened grid coordinates. N is the number of model grid
        points, m the number of coordinates dimensions (i.e. 2 or 3)
    points : numpy.ndarray, shape (p, m)
        array of POIs coordinates. P is the number of POIs, m the number of
        coordinates dimensions (i.e. 2 or 3)
    k : positive int
        number of considered nearest neighbors

    Returns
    -------
    dd : numpy.ndarray, shape (p, k)
        array of nearest neighbor distances.
    ii : numpy.ndarray, shape (p, k)
        array of nearest neighbor grid indexes

    See also
    --------
    :func:`nn_lookup_cached`
    """

    tree = KDTree(grid_coords)
    dd, ii = tree.query(points, k, workers=-1)

    return dd.astype(points.dtype, copy=False), ii


def nn_lookup_cached(
    grid_coords: np.ndarray, points: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Wrapper to :func:`nn_lookup` function to use it with a cache
    """

    hkey = (grid_coords.tobytes(), points.tobytes(), k)
    return nn_lookup(grid_coords, points, k, hkey=hkey)  # type: ignore[call-arg]


def _compute_dist_weight(dist: np.ndarray, power: float, dtype: np.dtype) -> np.ndarray:
    """
    Compute weights based on the inverse distance. The weights scales to 1.
    """

    increm = np.mean(dist, axis=1, keepdims=True) / 50  # avoid division by 0
    w = 1 / np.power(dist + increm, power)
    w = w.astype(dtype, copy=False)
    return w / np.sum(w, axis=1, keepdims=True)
