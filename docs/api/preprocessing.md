# Preprocessing: `pp`

```{eval-rst}
.. module:: napari_prism.pp
```

```{eval-rst}
.. currentmodule:: napari_prism
```

Filtering based on both marker intensity and cell-level observational values/quantiles. Common transforms used for marker intensity values. If backend is set to GPU (and configured), some functions wrap rapids_singlecell functions. Otherwise, same functions wrap scanpy.

## Functions

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   pp.set_backend
   pp.filter_by_obs_count
   pp.filter_by_obs_value
   pp.filter_by_obs_quantile
   pp.filter_by_var_value
   pp.filter_by_var_quantile
   pp.fill_na
   pp.log1p
   pp.arcsinh
   pp.zscore
   pp.scale
   pp.percentile
   pp.neighbors
   pp.add_obs_as_var
   pp.subset_adata_by_var
   pp.subset_adata_by_obs_category
```
