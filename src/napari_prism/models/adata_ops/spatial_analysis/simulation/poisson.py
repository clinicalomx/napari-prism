import numpy as np
from anndata import AnnData


def simulate_poisson_roi_data(
    ct_propotions, roi_radius, mean_num_cells, seed=None
):
    """
    Simulates a multitype 2D Poisson point process inside a circular region.

    By C.J.

    Parameters:
    - ct_proportions (list of float): Proportion of cell types. Length gives number of types
    - roi_radius (float): Radius of the circular domain (centered in the middle of the square).
    - mean_num_cells (float): Average number of cells in the ROI.
    - seed (int): Optional seed for reproducibility.

    Returns:
    - points (list of np.ndarray): List of arrays of shape (N_i, 2) for each type.
    - cell_types (list of strings): List of the type of each cell. Shape (N_i,).
    """
    if seed is not None:
        np.random.seed(seed)

    num_types = len(ct_propotions)
    width, height = 2 * roi_radius, 2 * roi_radius
    cx, cy = width / 2, height / 2

    area_circle = np.pi * (roi_radius**2)
    area_square = width * height
    ct_propotions = np.array(ct_propotions)
    rates = (
        ct_propotions * mean_num_cells
    )  # Expected number of cells in circle

    points = np.empty((0, 2))
    cell_types = np.empty(0)

    for i in range(num_types):
        # Sample number of points from Poisson
        n_expected = (
            rates[i] * area_square / area_circle
        )  # Scaling to account for rejection
        n_points = np.random.poisson(n_expected)

        # Rejection sampling: generate points in bounding square and filter to circle
        x = np.random.uniform(cx - roi_radius, cx + roi_radius, n_points)
        y = np.random.uniform(cy - roi_radius, cy + roi_radius, n_points)
        r2 = (x - cx) ** 2 + (y - cy) ** 2
        inside = r2 <= roi_radius**2
        accepted = np.column_stack((x[inside], y[inside]))
        points = np.vstack((points, accepted))
        cell_types = np.hstack(
            (cell_types, (np.repeat([f"{i}"], inside.sum())))
        )

    return points, cell_types


def simulate_poisson_roi_data_from_adata(
    adata: AnnData,
    roi_radius: int,
    spatial_key_added: str = "spatial",
    library_key: str | None = None,
    seed: int | None = None,
) -> AnnData:
    """
    Does same as `simulate_poisson_roi_data` but from existing single-cell
    AnnData object, where points (cells) already have pre-defined labels.

    Args:
        adata: Single cell level AnnData
        roi_radius: Radius of circular domain.
        library_key: Column in .obs that denotes distinct spatial regions or
            samples. If no key is provided, it assumes all cells are in the
            same physical space.
        seed: Optional seed for reproducibility.

    Returns:
        AnnData object with a new key, 'spatial' in .obsm denoting the
        simulated x, y spatial coordinates.

    """
    if seed is not None:
        np.random.seed(seed)

    width, height = 2 * roi_radius, 2 * roi_radius
    cx, cy = width / 2, height / 2

    all_coords = []
    all_cells = []

    if library_key is None:
        groups = [("", adata.obs)]
    else:
        groups = adata.obs.groupby(library_key)

    # Offset factor to keep ROIs non-overlapping
    offset_step = 3 * roi_radius

    for i, (_, obs_df) in enumerate(groups):
        n_points = len(obs_df)
        coords = np.empty((0, 2))

        while coords.shape[0] < n_points:
            n_sample = n_points * 2
            x = np.random.uniform(cx - roi_radius, cx + roi_radius, n_sample)
            y = np.random.uniform(cy - roi_radius, cy + roi_radius, n_sample)
            r2 = (x - cx) ** 2 + (y - cy) ** 2
            inside = r2 <= roi_radius**2
            accepted = np.column_stack((x[inside], y[inside]))
            coords = np.vstack((coords, accepted))

        coords = coords[:n_points]

        # Apply spatial offset to separate distinct libraries
        offset_x = i * offset_step
        coords[:, 0] += offset_x

        all_coords.append(coords)
        all_cells.append(obs_df.index.values)

    # Concatenate results
    all_coords = np.vstack(all_coords)

    # Store back into AnnData
    adata.obsm[spatial_key_added] = all_coords

    return adata


def homogeneous_gcross_theoretical_values(
    ct_propotions, roi_radius, mean_num_cells, gcross_radii
):
    """
    Calculates theoretical values of gcross for a homogeneous poisson process.

    By C.J.

    Parameters:
    - ct_proportions (list of float): Proportion of cell types. Length gives number of types
    - roi_radius (float): Radius of the circular domain (centered in the middle of the square).
    - mean_num_cells (float): Average number of cells in the ROI.
    - gcross_radii (list of float): Radii the gcross has been calculated at.

    Returns:
    - theoretical_values (np.array): Matrix of theoretical values for given process proportions
        and gcross radii. Shape is (num of gcross radii, num of cell types).
    """

    area_circle = np.pi * (roi_radius**2)
    ct_propotions = np.array(ct_propotions)
    unit_rates = (
        ct_propotions * mean_num_cells / area_circle
    )  # Expected number of cells in circle

    theoretical_value_exponent = (
        -np.expand_dims(unit_rates, axis=0)
        * (np.expand_dims(gcross_radii, axis=1) ** 2)
        * np.pi
    )
    theoretical_values = 1 - np.exp(theoretical_value_exponent)
    return theoretical_values
