# Plotting: `pl`

```{eval-rst}
.. module:: napari_prism.pl
```

```{eval-rst}
.. currentmodule:: napari_prism
```

Similar to scanpy, this module parallels some functions, but mainly for the
core `im.*` functions to display the results of TMA masking, dearraying and
segmentation. Some functions in `tl.*` also have a plotting function.

## Images

These functions wrap spatialdata_plot into scanpy-like functions to display
image, label and shapes data from a SpatialData object.

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   pl.image
   pl.mask_tma
   pl.dearray_tma
   pl.segment_tma
   pl.preview_tma_segmentation
```

## Tables

These functions display results from analyses involving AnnDatas / tables.

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   pl.cluster_scores
```
