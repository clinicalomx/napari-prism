"""
Tests for xarray-AnnData conversion functions in schema.py
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from anndata import AnnData

from napari_prism.models.adata_ops.schema import (
    CellEntity,
    CompartmentEntity,
    anndata_to_xarray,
    create_metric,
    xarray_to_anndata,
)


class TestSchemaConversion:
    """Test conversion between xarray and AnnData formats."""

    def test_simple_metric_conversion(self):
        """Test conversion of a simple spatial metric."""
        # Create test metric data
        cell_pop_a = CellEntity("T_cell", "cell_type")
        compartment_a = CompartmentEntity("nucleus", "compartment")

        values = np.array([1.5, 2.3, 0.8])
        sample_ids = ["sample_1", "sample_2", "sample_3"]

        # Create metric for each sample
        metrics = []
        for i, sample_id in enumerate(sample_ids):
            metric = create_metric(
                values=[values[i]],
                sample_id=sample_id,
                cell_population_a=cell_pop_a,
                cell_compartment_a=compartment_a,
                metric_name="density",
                parameters={"radius": 50},
            )
            metrics.append(metric)

        # Combine metrics
        combined_metric = xr.concat(metrics, dim="sample_id")

        # Convert to AnnData
        adata = xarray_to_anndata(combined_metric)

        # Verify structure
        assert adata.n_obs == 3
        assert adata.n_vars >= 1
        assert list(adata.obs.index) == sample_ids
        assert adata.X.shape == (3, adata.n_vars)

        # Check that metric parameters are stored
        assert "_xarray_dims" in adata.uns
        assert "_xarray_coords" in adata.uns
        assert "metric_name__parameters" in adata.uns

        # Convert back to xarray
        reconstructed = anndata_to_xarray(adata)

        # Verify reconstruction
        assert "sample_id" in reconstructed.dims
        assert len(reconstructed.coords["sample_id"]) == 3

    def test_pairwise_metric_conversion(self):
        """Test conversion of pairwise cell interaction metrics."""
        cell_pop_a = CellEntity("T_cell", "cell_type")
        cell_pop_b = CellEntity("B_cell", "cell_type")
        compartment_a = CompartmentEntity("nucleus", "compartment")
        compartment_b = CompartmentEntity("cytoplasm", "compartment")

        # Create pairwise interaction data - create separate metrics for each sample
        sample_ids = ["sample_1", "sample_2"]
        distance_bins = [10, 25, 50]

        metrics = []
        for sample_id in sample_ids:
            values = np.random.rand(3)  # 3 distance bins
            metric = create_metric(
                values=values,
                sample_id=sample_id,
                cell_population_a=cell_pop_a,
                cell_population_b=cell_pop_b,
                cell_compartment_a=compartment_a,
                cell_compartment_b=compartment_b,
                metric_name="interaction_score",
                dims=["distance"],
                coords={"distance": distance_bins},
                directional=True,
                parameters={"method": "ripley_k", "normalize": True},
            )
            metrics.append(metric)

        # Combine metrics along sample dimension
        metric = xr.concat(metrics, dim="sample_id")

        # Convert to AnnData
        adata = xarray_to_anndata(metric)

        # Verify pairwise structure
        assert adata.n_obs == 2
        assert adata.n_vars == 3  # 3 distance bins

        # Check feature metadata (values might be converted to strings)
        assert "distance" in adata.var.columns
        assert len(adata.var["distance"]) == len(distance_bins)

        # Check metric metadata
        assert "metric_name__directional" in adata.uns
        assert adata.uns["metric_name__directional"] is True
        assert "metric_name__parameters" in adata.uns

        # Convert back
        reconstructed = anndata_to_xarray(adata)

        # Verify dimensions are preserved
        assert "sample_id" in reconstructed.dims
        assert (
            "distance" in reconstructed.dims
            or "distance" in reconstructed.coords
        )

    def test_dataset_conversion(self):
        """Test conversion of xarray Dataset with multiple metrics."""
        cell_pop = CellEntity("T_cell", "cell_type")
        compartment = CompartmentEntity("nucleus", "compartment")

        sample_ids = ["sample_1", "sample_2"]

        # Create multiple metrics - one per sample, then combine
        density_metrics = []
        diversity_metrics = []

        for i, sample_id in enumerate(sample_ids):
            density_metric = create_metric(
                values=[1.5 + i * 0.8],  # Different values per sample
                sample_id=sample_id,
                cell_population_a=cell_pop,
                cell_compartment_a=compartment,
                metric_name="density",
            )
            density_metrics.append(density_metric)

            diversity_metric = create_metric(
                values=[0.8 + i * 0.1],  # Different values per sample
                sample_id=sample_id,
                cell_population_a=cell_pop,
                cell_compartment_a=compartment,
                metric_name="diversity",
            )
            diversity_metrics.append(diversity_metric)

        # Combine metrics along sample dimension
        combined_density = xr.concat(density_metrics, dim="sample_id")
        combined_diversity = xr.concat(diversity_metrics, dim="sample_id")

        # Create Dataset
        dataset = xr.Dataset(
            {"density": combined_density, "diversity": combined_diversity}
        )

        # Convert to AnnData
        adata = xarray_to_anndata(dataset)

        # Verify multiple metrics are combined
        assert adata.n_obs == 2
        # Number of features depends on how Dataset is flattened - should be at least 2
        assert adata.n_vars >= 2

        # Convert back (should create single DataArray with feature dimension)
        reconstructed = anndata_to_xarray(adata)
        assert "sample_id" in reconstructed.dims

    def test_roundtrip_conversion(self):
        """Test that data survives roundtrip conversion."""
        # Create test AnnData
        n_obs, n_vars = 5, 10
        X = np.random.rand(n_obs, n_vars)
        obs = pd.DataFrame(
            {
                "sample_type": ["A", "B", "A", "B", "A"],
                "batch": [1, 1, 2, 2, 2],
            },
            index=[f"sample_{i}" for i in range(n_obs)],
        )
        var = pd.DataFrame(
            {
                "feature_type": ["spatial"] * n_vars,
                "cell_population_a": ["T_cell"] * n_vars,
            },
            index=[f"feature_{i}" for i in range(n_vars)],
        )
        uns = {"method": "test", "parameters": {"param1": 1.0}}

        original_adata = AnnData(X=X, obs=obs, var=var, uns=uns)

        # Convert to xarray and back
        xarray_data = anndata_to_xarray(original_adata)
        reconstructed_adata = xarray_to_anndata(xarray_data)

        # Verify data integrity
        assert reconstructed_adata.n_obs == original_adata.n_obs
        assert reconstructed_adata.n_vars == original_adata.n_vars
        np.testing.assert_array_almost_equal(
            reconstructed_adata.X, original_adata.X
        )

        # Check sample IDs are preserved
        assert list(reconstructed_adata.obs.index) == list(
            original_adata.obs.index
        )

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with missing sample_id dimension
        da = xr.DataArray([1, 2, 3], dims=["feature"])
        with pytest.raises(ValueError, match="not found in data"):
            xarray_to_anndata(da)

        # Test with empty AnnData
        empty_adata = AnnData(X=np.array([]).reshape(0, 0))
        # Should not raise an error
        result = anndata_to_xarray(empty_adata)
        assert isinstance(result, xr.DataArray)

    def test_coordinate_attributes_preservation(self):
        """Test that coordinate attributes are preserved through conversion."""
        cell_pop = CellEntity("T_cell", "cell_type_column")
        compartment = CompartmentEntity("nucleus", "compartment_column")

        metric = create_metric(
            values=[1.5],
            sample_id="test_sample",
            cell_population_a=cell_pop,
            cell_compartment_a=compartment,
            metric_name="test_metric",
            parameters={"radius": 50, "method": "test"},
        )

        # Convert to AnnData
        adata = xarray_to_anndata(metric)

        # Check that entity attributes are preserved
        assert "cell_population_a__cell_type_column" in adata.uns
        assert (
            adata.uns["cell_population_a__cell_type_column"]
            == "cell_type_column"
        )
        assert "cell_compartment_a__compartment_column" in adata.uns

        # Convert back and verify attributes are restored
        reconstructed = anndata_to_xarray(adata)

        # Check coordinate attributes
        if "cell_population_a" in reconstructed.coords:
            attrs = reconstructed.coords["cell_population_a"].attrs
            assert "cell_type_column" in attrs
