import numpy as np

from napari_prism.models.adata_ops.spatial_analysis.simulation.poisson import (
    simulate_poisson_roi_data,
)


def test_simulate_poisson_roi_data():
    ct_proportions = np.array([0.3, 0.6, 1.0])
    spatial, cts = simulate_poisson_roi_data(ct_proportions, 1, 10000)
    spatial = spatial * 1000

    # 2d euclidean spatial
    assert spatial.shape[-1] == 2
    assert cts.shape[0] == spatial.shape[0]
    assert len(np.unique(cts)) == len(ct_proportions)


def test_simulate_poisson_roi_data_from_adata(adata_spatial):
    sp_key = "spatial_fake"
    assert sp_key in adata_spatial.obsm
    assert adata_spatial.obsm[sp_key].shape[1] == 2
