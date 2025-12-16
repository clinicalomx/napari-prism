import math
from functools import partial
from itertools import product
from math import pi
from pathlib import Path
from threading import Lock
from typing import Any, Literal

import anndata as ad
import geopandas
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import skimage
import spatialdata as sd
import torch
import xarray as xr
from anndata import AnnData
from dask.array import Array
from geopandas import GeoDataFrame
from loguru import logger
from numpy import dtype, float64, ndarray
from packaging import version
from scipy.ndimage import binary_fill_holes, generate_binary_structure
from shapely import Point, Polygon, geometry
from skimage import feature, morphology, transform
from sklearn.cluster import KMeans
from spatialdata import SpatialData
from spatialdata.models import (
    Image2DModel,
    Labels2DModel,
    ShapesModel,
    TableModel,
)
from spatialdata.models.models import Schema_t
from spatialdata.transformations import (
    BaseTransformation,
    Scale,
    Sequence,
    Translation,
    get_transformation,
    get_transformation_between_coordinate_systems,
    set_transformation,
)
from xarray import DataArray, DataTree

from napari_prism.constants import (
    CELL_INDEX_LABEL,
)

# For cellpose api
try:
    from torch import no_grad
except ImportError:

    def no_grad():
        def _deco(func):
            return func

        return _deco


__all__ = ["TMAMasker", "TMADearrayer", "TMASegmenter", "TMAMeasurer"]


class SdataImageOperations:
    """Base class for operating on Image elements in SpatialData objects.
    Contains general methods for image operations, such as contrast correction,
    projection, and coordinate system transformations."""

    #: Lock writing operations to one thread
    _write_lock = Lock()

    MULTICHANNEL_PROJECTION_METHODS = ["max", "mean", "sum", "median"]

    def __init__(
        self,
        sdata: SpatialData,
        image_name: str,
        reference_coordinate_system: str = "global",
    ) -> None:
        """
        Args:
            sdata: SpatialData object containing the image data.
            image_name: Name of the image element in the SpatialData object.
            reference_coordinate_system: The coordinate system to use for
                transformations and projections. Default is 'global' (pixels).
        """
        self.sdata = sdata
        self.image_name = image_name
        self.reference_coordinate_system = reference_coordinate_system

    def get_image(self) -> Any:
        """Get the image for masking given an index in the image pyramid to
        retrieve a certain resolution, as well as the provided channel."""
        raise NotImplementedError("calling from abstract method")

    def apply_image_correction(
        self,
        image: DataArray | ndarray[Any, dtype[float64]],
        contrast_limits: tuple[float, float],
        gamma: float = 1.0,
    ) -> DataArray | ndarray[Any, dtype[float64]]:
        """Apply contrast limits and gamma correction to an image.

        Args:
            image: The image to apply the correction to.
            contrast_limits: The minimum and maximum values for the contrast
                limits.
            gamma: The gamma value for gamma correction.
        Returns:
            The corrected image.
        """
        # NOTE: consider this a module func
        con_min, con_max = contrast_limits
        image = (image - con_min) / (con_max - con_min)

        if isinstance(image, DataArray):
            image = image.clip(0, con_max)
        else:
            np.clip(image, 0, con_max, out=image)
        image = image**gamma
        return image

    def get_transformation_to_cs(self, cs: str) -> BaseTransformation:
        """Convert current image to the chosen coordinate system.

        Args:
            cs: The coordinate system to convert to.

        Returns:
            The transformation to convert the contained image to the chosen
            coordinate system.
        """
        if cs not in self.sdata.coordinate_systems:
            raise ValueError(f"No {cs} coordinate system in sdata")

        return get_transformation_between_coordinate_systems(
            self.sdata, self.get_image(), cs
        )

    def get_unit_scaling_factor(
        self, coordinate_system: str
    ) -> BaseTransformation:
        """Get the scaling factor to convert pixel (global) units to the chosen
        coordinate system.

        Temporary implementation, following below for updates:
        - https://github.com/scverse/spatialdata/issues/30
        - https://github.com/scverse/spatialdata/issues/436

        Args:
            coordinate_system: The coordinate system to convert to.

        Returns:
            The scaling factor
        """
        # base_transformation = self.get_transformation_to_cs(coordinate_system)
        base_transformation = get_transformation_between_coordinate_systems(
            self.sdata,
            "global",  # Px
            coordinate_system,
            self.get_image(),  # as intermediate
        )
        scaling_factor_x = base_transformation.to_affine_matrix("x", "x")[0, 0]
        scaling_factor_y = base_transformation.to_affine_matrix("y", "y")[0, 0]
        assert (
            scaling_factor_x == scaling_factor_y
        ), "Unequal scaling factors for X and Y"
        return scaling_factor_x

    def get_px_per_um(self) -> float:
        """Get the number of pixels per micron in the current image.

        Returns:
            The number of pixels per micron in the curent image. Used for
            converting between pixel and micron units.
        """
        # Check if um is in the sdata
        if "um" not in self.sdata.coordinate_systems:
            raise ValueError("No um coordinate system in sdata")

        return 1 / self.get_unit_scaling_factor(
            coordinate_system="um"
        )  # px per micron

    def convert_um_to_px(self, um_value: int | float) -> int:
        """Convert a value from microns to pixels in the current image.

        Values are rounded to the nearest whole number since pixel values are
        integers. This is where 'rasterization' occurs, so rounding errors may
        occur here.

        Args:
            um_value: The value in microns to convert to pixels.

        Returns:
            The `um_value` equivalent in pixels.
        """
        px_value = np.round(um_value * self.get_px_per_um())
        return int(px_value)

    def get_multichannel_image_projection(
        self,
        image: DataArray,
        channels: list[str],
        method: Literal["max", "mean", "sum", "median"] = "max",
    ) -> DataArray:
        """Project a multichannel image to a single channel using a given
        method.

        Args:
            image: The multichannel image to project.
            channels: The channels to project.
            method: The method to use for projection. Can be 'max', 'mean',
                'sum', or 'median'.

        Returns:
            The projected image.
        """
        multichannel_selection = image.sel(c=channels)
        if method == "max":
            compacted = multichannel_selection.max("c", keep_attrs=True)
        elif method == "mean":
            compacted = multichannel_selection.mean("c", keep_attrs=True)
        elif method == "sum":
            compacted = multichannel_selection.sum("c", keep_attrs=True)
        elif method == "median":
            compacted = multichannel_selection.median("c", keep_attrs=True)
        else:
            raise ValueError("Invalid method for multichannel projection")

        # Add the c dim back for logging of what the slice represents
        c_name = f'{method}[{"_".join(channels)}]'  # + "_" + method
        # Keep the c dim as a scalar
        compacted = compacted.expand_dims(c=[1])
        compacted = compacted.assign_coords(c=[c_name])
        return compacted

    def overwrite_element(
        self, sdata: SpatialData, element: Schema_t, element_name: str
    ) -> None:
        """Overwrite an element in the SpatialData object with a new element.
        If the element already exists, it will be replaced.

        Args:
            sdata: The SpatialData object containing the element.
            element: The new element to add.
            element_name: The name of the element to add.
        """

        def _delete_from_disk(
            sdata: SpatialData, element_name: str, overwrite: bool
        ) -> None:
            if (
                element_name in sdata
                and len(sdata.locate_element(sdata[element_name])) != 0
            ):
                if overwrite:
                    with self._write_lock:
                        logger.info(f"Overwriting {element_name}")
                        del sdata[element_name]
                        sdata.delete_element_from_disk(element_name)
                else:
                    raise OSError(
                        f"`{element_name}` already exists. Use overwrite="
                        "True to rewrite."
                    )

        if sdata.is_backed():
            _delete_from_disk(sdata, element_name, overwrite=True)
        sdata[element_name] = element
        sdata.write_element(element_name, overwrite=True)

    def add_cs_from_model_image(
        self,
        element,
        query_cs="global",
        target_cs="um",
    ):
        """Adds the target coordinate system using the transformation
        of the given element to query, then appends and inherits this model's
        parent image transformation of query to target.

        i.e.) If `element` was derived from `parent`:
            element -- TransformA ---> query
            parent (query) -- TransformB ---> target
            Then element --- TransformA ---> query ---> TransformB ---> target
        """
        element_to_query = get_transformation(
            element, to_coordinate_system=query_cs
        )
        query_to_target = get_transformation_between_coordinate_systems(
            self.sdata, query_cs, target_cs, self.get_image()
        )
        element_to_target = Sequence([element_to_query, query_to_target])
        set_transformation(element, element_to_target, target_cs)

    def add_image(
        self,
        image: ndarray[Any, dtype[float64]] | DataArray | Array,
        image_label: str,
        write_element: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Adds a single or multiscale image to contained SpatialData.

        Args:
            image: The image to add.
            image_label: The name of the image to add.
            write_element: Whether to write the element to disk.
            *args: Passed to Image2DModel.parse.
            **kwargs: Passed to Image2DModel.parse.
        """
        image = Image2DModel.parse(image, *args, **kwargs)
        # new_image_name = f"{self.image_name}_{image_suffix}"
        if write_element:
            self.overwrite_element(self.sdata, image, image_label)
        else:
            self.sdata[image_label] = image
            logger.warning(
                "Spatialdata object is not stored on disk, could only add"
                " element in memory."
            )

        if "transformations" in kwargs:
            transformations = kwargs["transformations"]
            if "global" in transformations and "um" not in transformations:
                self.add_cs_from_model_image(
                    self.sdata[image_label], "global", "um"
                )
            elif "um" in transformations and "global" not in transformations:
                self.add_cs_from_model_image(
                    self.sdata[image_label], "um", "global"
                )

    def add_label(
        self,
        label: ndarray[Any, dtype[float64]] | DataArray | Array,
        label_name: str,
        write_element: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Adds single or multiscale image to contained SpatialData.

        Args:
            label: The label to add.
            label_name: The name of the label to add.
            write_element: Whether to write the element to disk.
            *args: Passed to Labels2DModel.parse.
            **kwargs: Passed to Labels2DModel.parse.
        """
        # Cast to dask -> DataArrays dont like inplace brush erasing / inpainting
        if isinstance(label, DataArray):
            label = label.data

        # NOTE
        # for compat with napari > 0.5, convert to uint8.
        # boolean arrays are checkd with _ensure_int_labels in labels.py
        # assumes data_level is a dask-like array;
        # it gets read as the DataTree, which has no view method;
        # so to skip this 'bug', ensure its np.uint8
        if label.dtype == bool:
            label = label.astype(np.uint8)

        label = Labels2DModel.parse(label, *args, **kwargs)
        # new_label_name = f"{self.image_name}_{label_suffix}"
        if write_element:
            self.overwrite_element(self.sdata, label, label_name)
        else:
            self.sdata[label_name] = label
            logger.warning(
                "Spatialdata object is not stored on disk, could only add"
                " element in memory."
            )
        if "transformations" in kwargs:
            transformations = kwargs["transformations"]
            if "global" in transformations and "um" not in transformations:
                self.add_cs_from_model_image(
                    self.sdata[label_name], "global", "um"
                )
            elif "um" in transformations and "global" not in transformations:
                self.add_cs_from_model_image(
                    self.sdata[label_name], "um", "global"
                )

    def add_shapes(
        self,
        shapes: GeoDataFrame,
        shapes_name: str | None = None,
        write_element: bool = False,
        # parent_table: TableModel | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Adds shapes to the contained SpatialData.

        Args:
            shapes: The shapes to add.
            shapes_name: The name of the shapes to add.
            write_element: Whether to write the element to disk.
            *args: Passed to ShapesModel.parse.
            **kwargs: Passed to ShapesModel.parse.
        """
        shapes = ShapesModel.parse(shapes, *args, **kwargs)

        if shapes_name is None:
            shapes_name = "shapes"

        if write_element:
            self.overwrite_element(self.sdata, shapes, shapes_name)
        else:
            self.sdata[shapes_name] = shapes
            logger.warning(
                "Spatialdata object is not stored on disk, could only add"
                " element in memory."
            )
        if "transformations" in kwargs:
            transformations = kwargs["transformations"]
            if "global" in transformations and "um" not in transformations:
                self.add_cs_from_model_image(
                    self.sdata[shapes_name], "global", "um"
                )
            elif "um" in transformations and "global" not in transformations:
                self.add_cs_from_model_image(
                    self.sdata[shapes_name], "um", "global"
                )

    def _get_scaled_polygon(self, polygon, scale) -> Polygon:
        """Returns the polygons, but re-scaled to 'real' world measurements."""
        scaled_coords = [
            (x * scale, y * scale) for x, y in polygon.exterior.coords
        ]
        return Polygon(scaled_coords)

    def _get_scaled_point(self, point, scale) -> Point:
        """Returns the circular polygons, but re-scaled to 'real' world
        measurements."""
        return Point(point.x * scale, point.y * scale)

    def add_table(
        self,
        table: AnnData | pd.DataFrame,
        table_suffix: str,
        write_element: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Adds a table to the contained SpatialData.

        Args:
            table: The table to add.
            table_suffix: The suffix to add to the table name.
            write_element: Whether to write the element to disk.
            *args: Passed to TableModel.parse.
            **kwargs: Passed to TableModel.parse.
        """
        if isinstance(table, pd.DataFrame):
            table = AnnData(obs=table)

        table = TableModel.parse(table, *args, **kwargs)
        new_table_name = f"{self.image_name}_{table_suffix}"

        if write_element:
            self.overwrite_element(self.sdata, table, new_table_name)
        else:
            self.sdata[new_table_name] = table
            logger.warning(
                "Spatialdata object is not stored on disk, could only add"
                " element in memory."
            )


class SingleScaleImageOperations(SdataImageOperations):
    """Base class for operating on single scale images in SpatialData objects
    (can labels elements, or single scale image elements)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_image(self) -> DataArray:
        """Get the image for masking given an index in the image pyramid to
        retrieve a certain resolution, as well as the provided channel.

        Returns:
            The contained image.
        """
        # ssi = self.sdata.image[self.image_name]
        ssi = self.sdata[self.image_name]  # Might be a .labels
        if not isinstance(ssi, DataArray):
            raise ValueError("Image is not a single scale image")
        return ssi

    def get_image_channels(self) -> ndarray:
        """Get the channel name of the image.

        Returns:
            The channel name of the image.
        """
        return self.get_image().coords["c"].data


class MultiScaleImageOperations(SdataImageOperations):
    """Base class for operating on multiscale images in SpatialData objects (
    multiscale image elements, DataArrays and DataTrees)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_image(self) -> DataTree:
        """Get the image pyramid/DataTree. Checks done here to ensure retrieved
        image is of the right type (Multiscale DataTree)"""
        msi = self.sdata[self.image_name]
        if not isinstance(msi, DataTree):
            raise ValueError("Image is not a multiscale image")
        return msi

    def get_image_flat(self, image: DataTree | None = None) -> list[DataArray]:
        """Return the image pyramid as a flat list of DataArrays.

        Args:
            image: The image to retrieve the flat list of DataArrays from. If
                None, uses the contained image in the instance.
        """
        if image is None:
            image = self.get_image()
        return [image[x].image for x in self.get_image_scales(image)]

    def get_image_channels(self, image: DataTree | None = None) -> ndarray:
        """Get the channel names of the contained image.

        Args:
            image: The image to retrieve the channels from. If None, uses the
                contained image in the instance.

        Returns:
            The channel names of the contained image.
        """
        return self.get_image_by_scale(image=image).coords["c"].data

    def get_image_scales(self, image: DataTree | None = None) -> list[str]:
        """Get the names of the image scales of the image pyramid. Removes the
        `/` prefix from the scale names.

        Args:
            image: The image to retrieve the scales from. If None, uses the
                contained image in the instance.

        Returns:
            The names of the image scales
        """
        if image is None:
            image = self.get_image()
        return [x.lstrip("/") for x in image.groups[1:]]

    def get_image_shapes(self, image: DataTree | None = None) -> list[str]:
        """Get the shapes of the images at each scale in the image pyramid.

        Usually in the form of: (C, Y, X) or (C, X, Y)

        Args:
            image: The image to retrieve the shapes from. If None, uses the
                contained image in the instance.

        Returns:
            The shapes of the images at each scale in the image pyramid.
        """
        scales = self.get_image_scales(image)
        if image is None:
            image = self.get_image()
        dataarrays = [image[x] for x in scales]
        shapes = [x.image.shape for x in dataarrays]
        return shapes

    def get_image_by_scale(
        self, image: DataTree | None = None, scale: str | None = None
    ) -> DataArray:
        """Get the image at a given scale in the image pyramid to
        retrieve a certain resolution.

        Args:
            image: The image to retrieve the scale from. If None, uses the
                contained image in the instance.
            scale: The scale of the image to retrieve. If None, retrieves
                the last scale (or smallest resolution) in the image pyramid.

        Returns:
            The image at the given scale.

        """
        if scale is None:
            scale = self.get_image_scales(image)[-1]  # smallest resolution

        if image is None:
            image = self.get_image()

        return image[scale].image

    def get_downsampling_factor(
        self,
        working_image: DataArray | ndarray[Any, dtype[float64]],
        source_image: DataArray | ndarray[Any, dtype[float64]] | None = None,
    ) -> float:
        """Get the downsampling factor which converts `working_image` to the
        full scale image. Only works if Y and X have equal scaling.

        Args:
            source_image: The source image to compute the full scale resolution
                from.
            working_image: The image to get the downsampling factor for.

        Returns:
            The downsampling factor.
        """
        fs = self.get_image_by_scale(source_image, scale="scale0")  # C,Y,X
        fs_x_shape = fs.coords["x"].shape[0]
        fs_y_shape = fs.coords["y"].shape[0]
        if isinstance(working_image, DataArray):
            ds_x_shape = working_image.coords["x"].shape[0]
            ds_y_shape = working_image.coords["y"].shape[0]
        else:
            ds_y_shape, ds_x_shape = working_image.shape
        ds_factor_x = fs_x_shape / ds_x_shape
        ds_factor_y = fs_y_shape / ds_y_shape

        # Technically these should be integers since the image is in pyramid format;
        # round to nearest whole number
        ds_factor_x = round(ds_factor_x)
        ds_factor_y = round(ds_factor_y)

        assert (
            ds_factor_x == ds_factor_y
        ), "Unequal downsampling factors for X and Y"
        return ds_factor_x


class TMAMasker(MultiScaleImageOperations):
    """Class for performing image masking operations on tissue microarray
    images."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mask_dataarray_image(
        self,
        image: DataArray | DataTree,  # For explicit coord
        xmin: int,
        ymin: int,
        xmax: int,
        ymax: int,
    ) -> DataArray:
        """Mask a DataArray image to have values only within the bounding box
        points. Values outside the bounding box are set to 0.

        Args:
            image: The image to mask.
            xmin: The minimum x-coordinate of the bounding box.
            ymin: The minimum y-coordinate of the bounding box.
            xmax: The maximum x-coordinate of the bounding box.
            ymax: The maximum y-coordinate of the bounding box.

        Returns:
            The masked image.
        """
        mask = np.zeros_like(image)
        mask[ymin:ymax, xmin:xmax] = True

        return mask * image

    def generate_blurred_masks(
        self,
        image: ndarray[Any, dtype[float64]] | DataArray | Array,
        sigma_px: int | float = 2,
        expansion_px: int | float = 3,
        li_threshold: bool = False,
        edge_filter: bool = False,
        adapt_hist: bool = False,
        gamma_correct: bool = False,
        fill_holes: bool = False,
        remove_small_spots_size: int | float = 0,
    ) -> ndarray[dtype[bool]]:
        """
        Generates simple 'segmentation' of TMA cores using piped
        operations of a (optional preprocessing) > gaussian blur >
        multiotsu > threshold > expansion.

        Args:
            image: The image to generate the masks from.
            sigma_px: The sigma value for the gaussian blur.
            expansion_px: The number of pixels to expand the mask by.
            li_threshold: Whether to use the Li thresholding method.
            edge_filter: Whether to use the Sobel edge filter.
            adapt_hist: Whether to use adaptive histogram equalization.
            gamma_correct: Whether to use gamma correction.

        Returns:
            The generated masks.
        """
        # image = image.astype("int64")
        # image = image.data.compute()
        image = image.data
        if li_threshold:
            th = skimage.filters.threshold_li(image)
            image = image * (image > th * 0.9)
        if edge_filter:
            image = skimage.filters.sobel(image)
        if adapt_hist:
            image = skimage.exposure.equalize_adapthist(image)
        if gamma_correct:
            image = skimage.exposure.adjust_gamma(image, gamma=0.2)

        blur = skimage.filters.gaussian(image, sigma=sigma_px)
        bts = skimage.filters.threshold_multiotsu(blur)[0]
        blur_thresholded = blur > bts
        # Expand mask by x pixels
        blur_thresholded_expanded = skimage.segmentation.expand_labels(
            blur_thresholded, expansion_px
        )

        # Additional morphological operators to enhance mask
        if fill_holes:
            footprint = generate_binary_structure(
                blur_thresholded_expanded.ndim, 1
            )

            blur_thresholded_expanded = binary_fill_holes(
                blur_thresholded_expanded, structure=footprint
            )

        if remove_small_spots_size > 0:  # skimage
            blur_thresholded_expanded = morphology.remove_small_objects(
                blur_thresholded_expanded, min_size=remove_small_spots_size
            )

        return blur_thresholded_expanded

    def generate_mask_contours(
        self, masks: ndarray[dtype[bool]], area_threshold: int | float = 0
    ) -> list[shapely.geometry.Polygon]:
        """Convert segmentation masks into contour formats (list of vertices),
        then filter contours by a pixel area threshold.

        Args:
            masks: The masks to generate contours from.
            area_threshold: The minimum area of a contour to keep.

        Returns:
            The filtered contours.
        """
        contours = skimage.measure.find_contours(
            masks.astype(np.int32), 0.5
        )  # Bool array;
        contours = [contour[:, [1, 0]] for contour in contours]
        contours = [shapely.Polygon(x) for x in contours]
        filtered_contours = []

        # Filter out very large areas;
        for c in contours:
            if c.area >= area_threshold:
                filtered_contours.append(c)

        return filtered_contours

    def estimate_threshold_from_radius(
        self,
        estimated_core_diameter_px: int | float,
        expansion_px: int | float = 3,
        core_fraction: int | float = 1,
    ) -> float:
        """Estimate a threshold for including whole or fractional circular
        objects of a given diameter in px.

        Args:
            estimated_core_diameter_px: The estimated diameter of the core in
                pixels.
            expansion_px: The number of pixels to expand the mask by.
            core_fraction: The fraction of the core to include.

        Returns:
            The estimated threshold.
        """
        estimated_core_radius_px = estimated_core_diameter_px / 2

        # Take into account mask expansion
        estimated_core_radius_px += expansion_px
        estimated_core_area_px2 = pi * (estimated_core_radius_px) ** 2
        return estimated_core_area_px2 * core_fraction

    def generate_mask_bounding_boxes(
        self, polygons: list[shapely.geometry.Polygon], bbox_pad: int = 0
    ) -> list[shapely.geometry.Polygon]:
        """Generates the unique/separated bounding boxes of a list of shapes
        (shapely.Polygon), with some padding.

        Args:
            polygons: The polygons to generate bounding boxes from.
            bbox_pad: The padding to apply to the bounding boxes.

        Returns:
            The bounding boxes.
        """
        # This draws bounding boxes around the TMA masks; also shapely polygons
        # with padding;
        print(bbox_pad)
        boxes = []
        for p in polygons:
            pad = (-bbox_pad, -bbox_pad, bbox_pad, bbox_pad)
            padded = np.array(p.bounds) + np.array(pad)
            boxes.append(shapely.geometry.box(*padded))

        return boxes

    def mask_tma_cores(
        self,
        channel: str | list[str],
        scale: str,
        mask_selection: tuple[int, int, int, int] | None = None,
        sigma_um: float | int = 10,
        expansion_um: float | int = 10,
        li_threshold: bool = False,
        edge_filter: bool = False,
        adapt_hist: bool = False,
        gamma_correct: bool = False,
        estimated_core_diameter_um: float | int = 700,
        core_fraction: float | int = 0.2,
        rasterize: bool = True,
        contrast_limits: tuple[float, float] = None,
        gamma: float = None,
        fill_holes: bool = False,
        remove_small_spots_size: int | float = 0,
    ) -> tuple[ndarray[dtype[bool]], BaseTransformation, str]:
        """Main masker function,

        1) Generate an initial mask using a gaussian blur
        2) Thresholds regions to only be above the area or fraction of an
            area above the estimated area (calculated from user-set core
            diameter estimate in mm)
        3) Generates unique bounding boxes which do not overlap. Overlapping
            regions get one merged bounding box.
        4) Results are saved to disk to the underlying sdata object.

        Args:
            channel: The channel to mask.
            scale: The scale of the image to mask.
            mask_selection: The bounding box (xmin, ymin, xmax, ymax ) to mask.
                If None, the whole image is masked.
            sigma_um: The sigma value for the gaussian blur in microns.
            expansion_um: The number of pixels to expand the mask by.
            li_threshold: Whether to use the Li thresholding method.
            edge_filter: Whether to use the Sobel edge filter.
            adapt_hist: Whether to use adaptive histogram equalization.
            gamma_correct: Whether to use gamma correction.
            estimated_core_diameter_um: The estimated diameter of the core in
                microns.
            core_fraction: The fraction of the core to include.
            rasterize: Whether to rasterize the masks.
            contrast_limits: The minimum and maximum values for the contrast
                limits.
            gamma: The gamma value for gamma correction.

        Returns:
            (Optional) If rasterize is False, the transformation sequence and
            the downsampling factor.
        """
        # Retrieve image to process
        multichannel_image = self.get_image_by_scale(scale=scale)  # C, Y, X

        selected_channel_image = multichannel_image.sel(c=channel)

        # Apply image correction if supplied;
        # if (contrast_limits is not None) and (gamma is not None):
        #     logger.info("Applying image correction")
        #     logger.info(f"contrast_limits : {contrast_limits}")
        #     logger.info(f"gamma : {gamma}")

        #     selected_channel_image = self.apply_image_correction(
        #         selected_channel_image, contrast_limits, gamma
        #     )

        # # Upscale transformations
        # upscale_transformations = Scale(
        #     [ds_factor, ds_factor], axes=("x", "y")
        # )
        # transformations.append(upscale_transformations)

        # # Subset to given bounding box --> Need to remap
        # if mask_selection is not None:
        #     xmin, ymin, xmax, ymax = mask_selection

        #     # Add translation to the transformations in the original space
        #     translation_transformation = Translation(
        #         [xmin, ymin], axes=("x", "y")
        #     )
        #     transformations.append(translation_transformation)

        #     # Remap bounding box to downsampled space
        #     xmin /= ds_factor
        #     ymin /= ds_factor
        #     xmax /= ds_factor
        #     ymax /= ds_factor
        #     xmin, ymin = map(math.floor, (xmin, ymin))
        #     xmax, ymax = map(math.ceil, (xmax, ymax))
        #     selected_channel_image = selected_channel_image.isel(
        #         x=slice(xmin, xmax), y=slice(ymin, ymax)
        #     )

        # transformation_sequence = Sequence(transformations)

        if mask_selection is not None:
            xmin, ymin, xmax, ymax = mask_selection
            selected_channel_image = sd.bounding_box_query(
                selected_channel_image,
                axes=("x", "y"),
                min_coordinate=[xmin, ymin],
                max_coordinate=[xmax, ymax],
                target_coordinate_system=self.reference_coordinate_system,
            )

        transformation_sequence = get_transformation(selected_channel_image)

        # convert um parameters to px equivalents;
        ds_factor = self.get_downsampling_factor(selected_channel_image)
        sigma_px = self.convert_um_to_px(sigma_um)
        expansion_px = self.convert_um_to_px(expansion_um)
        estimated_core_diameter_px = self.convert_um_to_px(
            estimated_core_diameter_um
        )

        # account for downsampling
        sigma_px /= ds_factor
        expansion_px /= ds_factor
        estimated_core_diameter_px /= ds_factor

        if isinstance(channel, list):
            # Perform a mask on each channel, sum the binary masks to perform a
            # 'merge'
            initial_masks = np.zeros(selected_channel_image.shape[1:])
            for c in channel:
                channel_mask = self.generate_blurred_masks(
                    selected_channel_image.sel(c=c),
                    sigma_px=sigma_px,
                    expansion_px=expansion_px,
                    li_threshold=li_threshold,
                    edge_filter=edge_filter,
                    adapt_hist=adapt_hist,
                    gamma_correct=gamma_correct,
                    fill_holes=fill_holes,
                    remove_small_spots_size=remove_small_spots_size,
                )
                initial_masks += channel_mask  # max intensity projection
                initial_masks = initial_masks > 0
                initial_masks = initial_masks.astype(np.int32)
        else:
            initial_masks = self.generate_blurred_masks(
                selected_channel_image,
                sigma_px=sigma_px,
                expansion_px=expansion_px,
                li_threshold=li_threshold,
                edge_filter=edge_filter,
                adapt_hist=adapt_hist,
                gamma_correct=gamma_correct,
                fill_holes=fill_holes,
                remove_small_spots_size=remove_small_spots_size,
            )

        # TODO: below is stilll quite slow, likely to do with transformation
        channel_label = (
            channel if not isinstance(channel, list) else "_".join(channel)
        )

        masks_gdf = None
        if rasterize:
            masks_gdf = self._rasterize_tma_masks(
                initial_masks,
                estimated_core_diameter_px,
                expansion_px,
                core_fraction,
            )

        return initial_masks, transformation_sequence, channel_label, masks_gdf

        # self.add_label(
        #     initial_masks,
        #     f"{channel_label}_mask",
        #     write_element=True,
        #     dims=("y", "x"),
        #     transformations={"global": transformation_sequence},
        # )

        # if rasterize:
        #     self._rasterize_tma_masks(
        #         initial_masks,
        #         estimated_core_diameter_px,
        #         expansion_px,
        #         core_fraction,
        #         channel_label,
        #         transformation_sequence,
        #     )
        # else:
        #     return transformation_sequence, ds_factor

    def save_shapes(
        self,
        masks_gdf: geopandas.GeoDataFrame,
        channel_label: str,
        transformation_sequence: BaseTransformation,
        write_element: bool = False,
    ):
        self.add_shapes(
            masks_gdf,
            f"{channel_label}_poly",
            write_element=write_element,
            transformations={
                self.reference_coordinate_system: transformation_sequence
            },
        )

    def save_masks(
        self,
        masks: ndarray[dtype[bool]],
        channel_label: str,
        transformation_sequence: BaseTransformation,
        write_element: bool = False,
    ):
        self.add_label(
            masks,
            f"{channel_label}",
            write_element=write_element,
            dims=("y", "x"),
            transformations={
                self.reference_coordinate_system: transformation_sequence
            },
        )

    def _rasterize_tma_masks(
        self,
        initial_masks: ndarray[dtype[bool]],
        estimated_core_diameter_px: int | float,
        expansion_px: int | float,
        core_fraction: int | float,
    ) -> None:
        """Rasterizes the masks generated by the mask_tma_cores function. Adds
        the rasterized masks to the SpatialData object.

        Args:
            initial_masks: The masks to rasterize.
            estimated_core_diameter_px: The estimated diameter of the core in
                pixels.
            expansion_px: The number of pixels to expand the mask by.
            core_fraction: The fraction of the core to include.
            channel_label: The label of the channel to rasterize.
            transformation_sequence: The transformation sequence to apply to
                the rasterized masks.
        """
        area_threshold = self.estimate_threshold_from_radius(
            estimated_core_diameter_px=estimated_core_diameter_px,
            expansion_px=expansion_px,
            core_fraction=core_fraction,
        )

        masks_polygons = self.generate_mask_contours(
            masks=initial_masks, area_threshold=area_threshold
        )

        masks_bboxes = self.generate_mask_bounding_boxes(
            polygons=masks_polygons
        )

        masks_gdf = self.consolidate_geometrical_objects(
            masks_polygons, masks_bboxes
        )

        # masks_table = TableModel.from_geodataframe(masks_gdf)
        return masks_gdf

    def consolidate_geometrical_objects(
        self,
        masks_polygons: list[shapely.geometry.Polygon],
        masks_bboxes: list[shapely.geometry.Polygon],
    ) -> geopandas.GeoDataFrame:
        """Consolidate the masks, polygons and bounding boxes into a single
        GeoDataFrame.

        The polygons are simplified with a tolerance = 2 to reduce the number of
        vertices.

        Args:
            masks_polygons: The polygons of the masks.
            masks_bboxes: The bounding boxes of the masks.

        Returns:
            The consolidated GeoDataFrame, with the polygons as the main
            `geometry` column and the bounding boxes an additonal column
            `masks_bboxes`.
        """
        # downsampling_factor = self.get_downsampling_factor(working_image)
        # scaling_func = partial(
        #     self._get_scaled_polygon, scale=1)

        masks_polygons_gs = geopandas.GeoSeries(masks_polygons)

        masks_bboxes_gs = geopandas.GeoSeries(masks_bboxes)

        agg = geopandas.GeoDataFrame(
            {"geometry": masks_polygons_gs, "masks_bboxes": masks_bboxes_gs},
            geometry="geometry",
        )

        # Reduce poly
        agg["geometry"] = agg["geometry"].simplify(tolerance=2)

        return agg


class TMADearrayer(SingleScaleImageOperations):
    """Class for dearraying tissue microarray images, but mainly on the mask
    outputs from TMAMasker."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #: The pixel positions of the estimated TMA grid in the image
        self.grid_positions = None

        #: The labels of the estimated TMA grid in the image
        self.grid_labels = None

        #: A GeoDataFrame containing each TMA core polygons representation
        self.core_gdf = None

        #: A GeoDataFrame containing the envelope/bounding box of each TMA core
        self.envelope_gdf = None

        #: A GeoDataFrame containing the merge of the core and envelope
        self.merged = None

    def estimate_rotation(
        self,
        blobs_dog: ndarray[Any, dtype[float64]],
        expected_diameter_px: int | float,
    ) -> float:
        """Estimates the rotation in the image from a list of blobs
        generated from skimage blobs dog detection.

        Does this in 2 steps:
        1. Detect blobs which are in the same row by checking if they are
            separated by atleast 2x the expected diameter in the x-axis,
            but are within the expected diameter in the y-axis. Store all
            these angles across the entire image.
        2. Estimate the angle of rotation of rows by taking the median of
            the stored angles.

        Args:
            blobs_dog: The blobs generated from skimage blobs dog detection.
            expected_diameter_px: The expected diameter of the blobs in pixels.

        Returns:
            The estimated rotation angle in degrees.
        """
        angles = []
        for blobs in blobs_dog:
            y, x, _ = blobs

            for blobs_next in blobs_dog:
                y_next, x_next, _ = blobs_next

                if (
                    (x_next > x)
                    and (x_next - x) > expected_diameter_px * 2
                    and abs(y_next - y) < expected_diameter_px
                ):
                    angle = (180 / math.pi) * math.atan2(
                        y_next - y, x_next - x
                    )
                    angles.append(angle)

        if len(angles) == 0:
            return 0

        return np.median(angles)

    def detect_blobs(
        self,
        image: ndarray[Any, dtype[float64]] | DataArray | Array,
        expected_diameter_um: int | float,
        expectation_margin: int | float,
    ) -> tuple[
        ndarray[Any, dtype[float64]], float, ndarray[Any, dtype[float64]]
    ]:
        """Detect circular blobs of a given radii within some expectation
        margin. Expectation margin is essentially the min and max sigma
        values for DoG detection.

        Args:
            image: The image to detect blobs in.
            expected_diameter_um: The expected diameter of the blobs in microns.
            expectation_margin: The margin of the expected diameter.

        Returns:
            The detected blobs, the estimated rotation angle, and the image
            centroid.
        """
        expected_radius_um = expected_diameter_um / 2
        expected_radius_px = self.convert_um_to_px(expected_radius_um)
        logger.info(f"estimated_core_radius_um : {expected_radius_um}")
        logger.info(f"estimated_core_radius_px : {expected_radius_px}")

        # Account for the image's transform;
        decomposed = self.get_image().transform["global"].transformations
        scales = [x for x in decomposed if isinstance(x, Scale)]
        if scales != []:  # or len 0
            scale = scales[0].scale[0]
            logger.info(
                f"working with image with downsampling factor : {scale}"
            )
            expected_radius_px /= scale
            logger.info(f"adjusted expected_radius_px : {expected_radius_px}")

        # Accounts for incorrect diameter estimates
        min_sigma = expected_radius_px * (1 - expectation_margin)
        max_sigma = expected_radius_px * (1 + expectation_margin)

        # Add approximation; 1 px radius ~ sqrt(2) sigma
        denom = np.sqrt(2)
        min_sigma = min_sigma / denom
        max_sigma = max_sigma / denom

        # Blob detection with blob_dog
        blobs_dog = feature.blob_dog(
            image,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            threshold=0.1,
            overlap=0.5,
        )
        blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)

        # Rotation correction
        angle = self.estimate_rotation(blobs_dog, expected_radius_px * 2)
        logger.info(f"estimated rotation angle : {angle}")
        # Extract centroid for shapely affine rotations as well
        rows, cols = image.shape[0], image.shape[1]
        image_centroid = np.array((rows, cols)) / 2.0 - 0.5

        blurred_image_rotated = transform.rotate(
            image, angle=angle, resize=False, center=image_centroid
        )  # As above, just so its in (rows, cols) format

        # Repeat;
        blobs_dog = feature.blob_dog(
            blurred_image_rotated,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            threshold=0.1,
            overlap=0.5,
        )
        blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)  # Correct radii for DoG
        return (
            blobs_dog,
            angle,
            image_centroid,
        )  # Also return the metadata for processing

    def estimate_grid(
        self,
        blobs_dog: ndarray[Any, dtype[float64]],
        nrows: int = 0,
        ncols: int = 0,
    ) -> tuple[ndarray[Any, dtype[float64]], ndarray[Any, dtype[float64]]]:
        """
        Simple grid estimation algoriithm.

        Estimate and create a grid layout model from the position of the
        blobs in blobs_dog. Uses either of the following:

        1) A traversal algorithm to segment the points into rows and columns.

        2) A Kmeans algorithm to segment the points into an expected number
        of rows and columns. Uses the kmeans algorithm to segment the points
        into rows and columns.

        From the expected row and column coordiantes, constructs an ideal /
        perfect grid, and the coordinates of each grid point.

        Args:
            blobs_dog: The blobs detected in the image.
            nrows: The expected number of rows.
            ncols: The expected number of columns.

        Returns:
            The grid positions and the grid labels.
        """

        def _walk_and_segment_points(sorted_points, radius):
            """Estimate the number of 'distinct' intervals or segments in
            sorted_points, by walking through each point, taking into
            account an estimated radius separating each point.

            Should take in suggested xs and ys.
            """
            previous_point = 0
            row_list = []
            current_list = []
            for i, p in enumerate(sorted_points):
                if i == 0:  # First point
                    previous_point = p
                    row_list.append(p)
                elif (
                    i == len(sorted_points) - 1
                ):  # Last point (most bottom / right)
                    row_mean = np.mean(row_list)
                    current_list.append(row_mean)
                else:
                    if (
                        p - previous_point < radius
                    ):  # If the position of the pt is within a radius dist,
                        row_list.append(p)
                    else:
                        row_mean = np.mean(
                            row_list
                        )  # The next pt indicates there's a new row/col
                        current_list.append(
                            row_mean
                        )  # use that as the avg row/col pos
                        row_list = []
                        row_list.append(p)
                    previous_point = p
            return current_list

        def _label_elements(N, M):
            """Label each element in the grid according to a standard TMA
            core formatting.

            Rows: Capital Alphabet [A-Z],
            Columns: Numerics [0-9]*
            Formatting: [A-Z]-[0-9]*
            """
            # Initialize an empty array for the labels
            labels = np.empty((N, M), dtype="object")

            for i in range(N):  # For each row
                row_label = chr(65 + i)  # Convert row index to alphabet (A-Z)
                for j in range(M):  # For each column
                    # Combine row label and column number (starting from 1)
                    labels[i, j] = f"{row_label}-{j + 1}"
            return labels

        def _kmeans_segment_points(points, n_expected):
            km = KMeans(n_expected)
            km.fit_predict(np.array(points).reshape(-1, 1))
            return [x[0] for x in sorted(km.cluster_centers_)]

        blobs_df = pd.DataFrame(blobs_dog, columns=["x", "y", "r"])
        mean_radius = blobs_df["r"].mean()
        x_sorted = blobs_df["x"].sort_values().values
        y_sorted = blobs_df["y"].sort_values().values

        if nrows > 0:
            current_x_list = _kmeans_segment_points(x_sorted, nrows)
        else:
            current_x_list = _walk_and_segment_points(x_sorted, mean_radius)

        if ncols > 0:
            current_y_list = _kmeans_segment_points(y_sorted, ncols)
        else:
            current_y_list = _walk_and_segment_points(y_sorted, mean_radius)

        combinations = list(product(current_x_list, current_y_list))
        # Arrange the combinations into a matrix of shape (len(x), len(y), 2);
        # last dim is x,y coord
        grid_positions = np.array(
            [
                combinations[
                    i * len(current_y_list) : (i + 1) * len(current_y_list)
                ]
                for i in range(len(current_x_list))
            ]
        )
        grid_positions = grid_positions.transpose(0, 1, 2)
        grid_labels = _label_elements(len(current_x_list), len(current_y_list))
        return grid_positions, grid_labels

    # TODO: below if we want to do thin plate spline estimation;
    # def estimate_grid_tps(
    #     self,
    #     blobs_dog: ndarray[Any, dtype[float64]],
    #     nrows: int = 0,
    #     ncols: int = 0,
    # ) -> tuple[ndarray[Any, dtype[float64]], ndarray[Any, dtype[float64]]]:
    #     """ """

    def dearray(
        self,
        expected_diameter_um: int | float,
        expectation_margin: int | float = 0.2,
        expected_rows: int = 0,
        expected_cols: int = 0,
    ) -> GeoDataFrame:
        """Dearray the TMA cores using the image or the mask generated by
        TMAMasker.

        Resolves the grid layout from the initial detection and grid
        estimation results using structured data formats (GeoDataFrames)

        Args:
            expected_diameter_um: The expected diameter of the cores in microns.
            expectation_margin: The margin of the expected diameter.
            expected_rows: The expected number of rows.
            expected_cols: The expected number of columns.

        Returns:
            The GeoDataFrame containing the dearrayed TMA cores. Cores are
                represented as Point objects with a radius (x, y, r).
        """
        image = self.get_image()  # Allow for multiple images;
        self.transforms = image.transform  # Inherit transforms from the image
        image = image.astype(bool)  # enforce binary
        # convert um to px
        # expected_diameter_px = self.convert_um_to_px(expected_diameter_um)

        cores_circles, image_rotation, image_centroid = self.detect_blobs(
            image, expected_diameter_um, expectation_margin
        )

        # image_rotation *= -1

        grid_positions, grid_labels = self.estimate_grid(
            cores_circles, expected_rows, expected_cols
        )

        self.grid_positions = grid_positions
        self.grid_labels = grid_labels

        # Create structured data for above
        cores_gdf = gpd.GeoDataFrame(pd.DataFrame(cores_circles))
        cores_gdf.columns = ["y", "x", "radius"]
        cores_gdf["circles"] = cores_gdf.apply(
            lambda x: Point(x["x"], x["y"]).buffer(x["radius"]), axis=1
        )
        cores_gdf = cores_gdf.set_geometry("circles")
        MISSING_LABEL = ""
        final_grid_labels = grid_labels.copy()
        radii = cores_gdf["radius"].mean()
        radii_threshold = radii * 1.25
        cores_gdf["tma_label"] = MISSING_LABEL

        # Iterate through each grid coordinate
        for i in range(grid_positions.shape[0]):
            for j in range(grid_positions.shape[1]):
                grid_point = Point(
                    grid_positions[i, j][::-1]
                )  # Get the current grid point to test

                # Get the closest circle -> THis can be erronous;
                closest_idx = cores_gdf.distance(grid_point).idxmin()
                closest = cores_gdf.loc[closest_idx]["circles"]
                # if the circle centroid is radius away from the grid point
                circle_found = closest.distance(grid_point) <= radii_threshold

                if circle_found:
                    # Check if observed circle has been assigned already
                    assigned_label = cores_gdf.loc[closest_idx, "tma_label"]
                    if assigned_label != MISSING_LABEL:
                        p_i, p_j = np.where(grid_labels == assigned_label)
                        p_i, p_j = p_i[0], p_j[0]
                        previous_point = Point(grid_positions[p_i, p_j][::-1])
                        # Check which is closer
                        if closest.distance(previous_point) < closest.distance(
                            grid_point
                        ):
                            # make this missing
                            final_grid_labels[i, j] = MISSING_LABEL
                            continue
                        else:
                            # Make the previous missing
                            final_grid_labels[p_i, p_j] = MISSING_LABEL

                    # Assign the closest circle to that label
                    cores_gdf.loc[closest_idx, "tma_label"] = grid_labels[i, j]
                    cores_gdf.loc[closest_idx, "tma_perfect_position_y"] = (
                        grid_positions[i, j][0]
                    )
                    cores_gdf.loc[closest_idx, "tma_perfect_position_x"] = (
                        grid_positions[i, j][1]
                    )
                else:
                    # Assign a missing label
                    final_grid_labels[i, j] = MISSING_LABEL

        self.final_grid_labels = final_grid_labels
        # Then check if any circles are intersecting;
        # Spatial join of self; sql like interesction operation of self;
        local_gdf = cores_gdf.copy()
        local_gdf = local_gdf.set_geometry("circles")
        joined_gdf = gpd.sjoin(
            local_gdf.reset_index(),
            local_gdf,
            how="inner",
            predicate="intersects",
        )
        intersections = joined_gdf[
            joined_gdf["index"] != joined_gdf["index_right"]
        ]

        if len(intersections) > 0:
            print("Intersecting cores found, going with 1.15 of diameter")
            # limit search to 5 iterations
            return self.dearray(
                expected_diameter_um * 1.15,
                expectation_margin,
                expected_rows,
                expected_cols,
            )

            # raise ValueError(
            #         "Intersecting cores found. " \
            #         "Try reducing the estimated core diameter. ")

        # Then final checks for missing values;
        # if any(cores_gdf["tma_label"] == MISSING_LABEL):
        #     raise ValueError(
        #             "Circles with unassigned values after checks. " \
        #             "Try changing the estimated core diameter. ")

        # Then unrotate circle coordinates;
        cores_gdf["circles"] = cores_gdf["circles"].rotate(
            angle=image_rotation, origin=Point(image_centroid)
        )

        # Cache bounding box of each circle
        cores_gdf["circles_bboxes"] = cores_gdf["circles"].map(
            lambda x: geometry.box(
                x.bounds[0], x.bounds[1], x.bounds[2], x.bounds[3]
            )
        )

        # Duplicate repr of points -> TODO: double check if needed or not
        cores_gdf["point"] = cores_gdf[["x", "y"]].apply(
            lambda x: Point(*x), axis=1
        )
        cores_gdf = cores_gdf.set_geometry(
            "point"
        )  # technically our core repr are Points -> Buffered to be 'circles'
        cores_gdf["point"] = cores_gdf["point"].rotate(
            angle=image_rotation, origin=Point(image_centroid)
        )
        cores_gdf["x"] = cores_gdf["point"].map(lambda c: c.x)
        cores_gdf["y"] = cores_gdf["point"].map(lambda c: c.y)
        cores_gdf = cores_gdf.set_geometry("circles")

        return cores_gdf

    def _generate_enveloping_bounding_boxes(
        self, core_gdf: GeoDataFrame, masks_gdf: GeoDataFrame | None = None
    ) -> None:
        """Merges the bounding boxes of the TMA masks (data transfer from
        core_gdf, provided by controller/viewer/presenter class) and the
        TMA core to get an all-enveloping bounding box.

        If `masks_gdf` is None, then just uses the cores.

        If provided, then gets the envelope of the mask + core.

        Writes the merged GeoDataFrame as a table to the underlying sdata
        object.

        Args:
            core_gdf: The GeoDataFrame containing the TMA cores.
            masks_gdf: The GeoDataFrame containing the TMA masks.
        """
        scaling_func = partial(self._get_scaled_polygon, scale=1)

        # Scale up polygon attributes
        core_gdf["circles"] = core_gdf["circles"].map(
            lambda p: scaling_func(p)
        )
        core_gdf["circles_bboxes"] = core_gdf["circles_bboxes"].map(
            lambda p: scaling_func(p)
        )
        core_gdf["point"] = core_gdf["point"].map(
            lambda p: self._get_scaled_point(p, 1)
        )

        core_gdf.set_geometry("circles_bboxes", inplace=True)

        # Merge super and current geopandasdfs
        if masks_gdf is None:
            masks_gdf = core_gdf

        # Join based on intersecting bounding boxes;
        masks_gdf.set_geometry("masks_bboxes", inplace=True)

        intersections = masks_gdf.sjoin(
            core_gdf,  # Will have duplicates. core:mask -> 1:N
            how="left",
            predicate="intersects",
        )
        merged = intersections.merge(
            core_gdf.reset_index()[["index", "circles_bboxes"]],
            left_on="index_right",
            right_on="index",
            how="left",
        )
        # Dictionary to hold the merged geometries for each unique circle_bbox
        merged_geometries = {}

        # Iterate over unique circle_bboxes, excluding None values
        for circle_bbox in merged["circles_bboxes"].dropna().unique():
            # Filter rows that intersect with the current circle_bbox
            intersecting_rows = []

            for mask_bbox in merged["masks_bboxes"]:
                if mask_bbox is not None and mask_bbox.intersects(circle_bbox):
                    intersecting_rows.append(mask_bbox)

            # If there are any intersections, merge them
            if intersecting_rows:
                all_geometries = intersecting_rows
                all_geometries.append(circle_bbox)
                unified_geometry = gpd.GeoSeries(all_geometries).unary_union
                new_box = geometry.box(*unified_geometry.bounds)
                merged_geometries[circle_bbox.wkt] = new_box
            else:
                # If no intersections, use the circle_bbox itself
                merged_geometries[circle_bbox.wkt] = circle_bbox

        # Map the merged geometries back to the original DataFrame
        # Handle None values explicitly
        merged["merged_geometry"] = merged["circles_bboxes"].apply(
            lambda x: merged_geometries[x.wkt] if x is not None else None
        )

        # TEMP for debuging
        # self.merged = merged
        # set of polygon relationally consistent 'database' of polygons
        # By default the TMA masks without core have been dropped in the joins
        # But, clean up final outputs
        # Overwrite some objects in sdata / viewer rep
        tma_masks = merged[["geometry", "masks_bboxes", "tma_label"]].rename(
            columns={"tma_masks": "geometry"}
        )  # duplicate tmas due to multiple masks in one core
        tma_masks = tma_masks.set_geometry("geometry")
        self.masks_bboxes = tma_masks["masks_bboxes"]

        tma_cores = core_gdf[
            ["point", "x", "y", "radius", "circles_bboxes", "tma_label"]
        ].rename(columns={"point": "geometry"})
        tma_cores = tma_cores.set_geometry("geometry")

        self.core_gdf = tma_cores

        # The enveloping box which encompasses the mask AND core bboxes.
        # useful for seg tiles
        tma_envelope = merged[["merged_geometry", "tma_label"]].rename(
            columns={"merged_geometry": "geometry"}
        )
        tma_envelope = tma_envelope.dropna().set_geometry("geometry")
        tma_envelope = tma_envelope.drop_duplicates("tma_label")
        tma_envelope = tma_envelope.reset_index(drop=True)
        self.envelope_gdf = tma_envelope  # TODO: resolve cores, but no masks.

        # Table that annotates these objects.
        cores = AnnData(obs=pd.DataFrame(tma_cores["tma_label"]))
        cores.obs["lyr"] = "tma_core"

        envelopes = AnnData(obs=pd.DataFrame(tma_envelope["tma_label"]))
        # envelopes.obs["image"] = self.image_name
        envelopes.obs["tma_label"] = envelopes.obs["tma_label"].astype("str")
        envelopes.obs["instance_id"] = tma_envelope.index
        envelopes.obs["lyr"] = (
            "tma_envelope"  # region; match the shapes element name above
        )
        envelopes.obs["lyr"] = envelopes.obs["lyr"].astype("category")
        envelopes.uns["grid_labels"] = self.grid_labels
        envelopes.uns["grid_positions"] = self.grid_positions

        # cores.obs["lyr_lbl"] = self.image_name + "_tma_core"
        # TODO: Need to drop the geoms; these are added as shapesmodel above

        # self.add_table(
        #     cores,
        #     "tma_core_tbl",
        #     write_element=True,
        #     region_key="lyr",
        #     region=self.image_name + "_tma_core",
        #     instance_key="tma_label"
        # )
        envelopes.uns["grouping_factor"] = "tma_label"
        envelopes.obs = envelopes.obs.set_index("tma_label")
        envelopes.obs["tma_label_col"] = envelopes.obs.index

        return tma_cores, tma_envelope, envelopes, self.transforms

    def save_dearray_results(
        self, tma_cores, tma_envelope, envelopes, transforms=None
    ):
        if transforms is None:
            transforms = self.transforms
        self.add_shapes(
            tma_cores,
            "tma_core",
            write_element=False,
            transformations=transforms,
        )
        self.add_shapes(
            tma_envelope,
            "tma_envelope",
            write_element=False,
            transformations=transforms,
        )

        self.add_table(
            envelopes,
            "tma_table",
            write_element=False,
            region="tma_envelope",  # element name
            region_key="lyr",  # region
            instance_key="instance_id",
        )

    def dearray_and_envelope_tma_cores(
        self,
        expected_diameter_um,
        expectation_margin=0.2,
        expected_rows=0,
        expected_cols=0,
        masks_gdf=None,
    ) -> tuple[GeoDataFrame, GeoDataFrame, AnnData, BaseTransformation]:
        """Main dearray function,

        1) Perform the dearrayer.
        2) Merge and resolve intersections between the TMA masks and TMA
            cores to get a envelope. Useful for getting full image subsets
            of each core automatically.

        Writes the final GeoDataFrames to the underlying sdata object.

        Args:
            expected_diameter_um: The expected diameter of the cores in microns.
            expectation_margin: The margin of the expected diameter.
            expected_rows: The expected number of rows.
            expected_cols: The expected number of columns.
            masks_gdf: The GeoDataFrame containing the TMA masks.

        """
        core_gdf = self.dearray(
            expected_diameter_um,
            expectation_margin,
            expected_rows,
            expected_cols,
        )
        results = self._generate_enveloping_bounding_boxes(core_gdf, masks_gdf)
        return results  # tma_cores, tma_envelope, envelopes, self.transforms

    # TODO: user interaction for manual grid modifications
    def append_tma_row(self) -> None:
        """Append a new row to the TMA grid representation."""
        grid_labels = self.grid_labels
        nrows, ncols = grid_labels.shape
        new_row_label = chr(
            65 + nrows
        )  # py 0 indexing, but shape is counted from 1
        new_row = [{new_row_label} - {j + 1} for j in range(ncols)]
        grid_labels = np.vstack([grid_labels, new_row])
        self.grid_labels = grid_labels

    def remove_tma_row(self, row_index) -> None:
        """Remove a row from the TMA grid representation."""
        raise NotImplementedError("Not implemented yet")

    def append_tma_column(self) -> None:
        """Append a new column to the TMA grid representation."""
        raise NotImplementedError("Not implemented yet")

    def remove_tma_column(self, column_index) -> None:
        """Remove a column from the TMA grid representation."""
        raise NotImplementedError("Not implemented yet")

    def add_tma_core(self, geometry, row_index, column_index) -> None:
        """Add a TMA core to the TMA grid representation."""
        raise NotImplementedError("Not implemented yet")

    def remove_tma_core(self, row_index, column_index) -> None:
        """Remove a TMA core from the TMA grid representation."""
        raise NotImplementedError("Not implemented yet")

    # Below operate on the relational df
    # def _check_tma_df_exists(self):
    #     if self.merged is None:
    #         raise AttributeError(
    #             "Run initial dearray first."
    #         )  # TODO: the user should be able to add first anway

    # def add_tma_core(self, geometry) -> None:
    #     """Add a TMA core to the TMA grid representation."""
    #     self._check_tma_df_exists()
    #     # add tma cores to model

    # def remove_tma_core(self) -> None:
    #     """Remove a TMA core from the TMA grid representation."""
    #     self._check_tma_df_exists()
    #     # remove core entry from list


class TMASegmenter(MultiScaleImageOperations):
    """Class for performing cell segmention tissue microarray images. Currently,
    only supports cellpose."""

    #: Default cellpose 3.0 models.
    CP_DEFAULT_MODELS = [
        "cyto3",
        "cyto2",
        "cyto",
        "nuclei",
        "tissuenet_cp3",
        "livecell_cp3",
        "yeast_PhC_cp3",
        "yeast_BF_cp3",
        "bact_phase_cp3",
        "bact_fluor_cp3",
        "deepbacs_cp3",
        "cyto2_cp3",
    ]

    #: Default cellpose 3.0 denoiser models.
    CP_DENOISE_MODELS = [
        "nan",
        "denoise_cyto3",
        "deblur_cyto3",
        "upsample_cyto3",
        "oneclick_cyto3",
        "denoise_cyto2",
        "deblur_cyto2",
        "upsample_cyto2",
        "oneclick_cyto2",
        "denoise_nuclei",
        "deblur_nuclei",
        "upsample_nuclei",
        "oneclick_nuclei",
    ]

    #: Typed versions of the above lists.
    CP_DEFAULT_MODELS_typed = Literal[
        "cyto3",
        "cyto2",
        "cyto",
        "nuclei",
        "tissuenet_cp3",
        "livecell_cp3",
        "yeast_PhC_cp3",
        "yeast_BF_cp3",
        "bact_phase_cp3",
        "bact_fluor_cp3",
        "deepbacs_cp3",
        "cyto2_cp3",
    ]

    CP_DENOISE_MODELS_typed = Literal[
        "nan",
        "denoise_cyto3",
        "deblur_cyto3",
        "upsample_cyto3",
        "oneclick_cyto3",
        "denoise_cyto2",
        "deblur_cyto2",
        "upsample_cyto2",
        "oneclick_cyto2",
        "denoise_nuclei",
        "deblur_nuclei",
        "upsample_nuclei",
        "oneclick_nuclei",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_cpsam = False

    @no_grad()
    def cellpose_segmentation(
        self,
        image: (
            ndarray[Any, dtype[float64]] | list[ndarray[Any, dtype[float64]]]
        ),
        model_type: CP_DEFAULT_MODELS_typed,
        channels: list[int] | list[list[int]],
        channel_axis: int = 2,
        nuclei_diam_um: float | None = None,
        normalize: bool = True,
        cellprob_threshold: float = 0.0,
        flow_threshold: float = 0.4,
        custom_model: Path | str | bool | None = False,
        denoise_model: CP_DENOISE_MODELS_typed | None = None,
        **kwargs,
    ):
        """Wrapper for building and running cellpose models.

        Update: Dynamically checking if Cellpose3, or newer SAM implementation.

        Args:
            image: The image to segment.
            model_type: The type of cellpose model to use.
            channels: The channels to segment.
            channel_axis: The axis of the channels.
            nuclei_diam_um: The diameter of the nuclei in microns. If None,
                then the diameter is automatically estimated.
            normalize: If True, does image intensity normalization.
            cellprob_threshold: The cell probability threshold.
            flow_threshold: The flow threshold.
            custom_model: The custom model to use.
            denoise_model: The denoiser model to use.

        Returns:
            Dictionary of the segmentation results and metadata
        """
        import cellpose.version
        from cellpose import core, denoise, models

        device = None
        # sentinel values for diameter
        if nuclei_diam_um is not None:
            if nuclei_diam_um <= 0:
                nuclei_diam_px = None  # Automated diameter estimation
            else:
                nuclei_diam_px = self.convert_um_to_px(
                    nuclei_diam_um
                )  #  * self.get_px_per_um() # px
        else:
            nuclei_diam_px = None

        gpu = core.use_gpu()

        # Check if macos + mps
        if torch.backends.mps.is_available():  # noqa: F821
            gpu = True
            device = torch.device("mps")  # noqa: F821
            logger.info("Using MPS")

        # TODO: refine these checks
        if custom_model is None:
            custom_model = False

        self.is_cpsam = version.parse(cellpose.version) >= version.parse("4")
        if custom_model is False and self.is_cpsam:
            print("Using CellposeSAM")
            custom_model = "cpsam"

        if denoise_model == "nan":
            denoise_model = None

        if denoise_model is None:
            denoise_model = False

        if denoise_model:
            model = denoise.CellposeDenoiseModel(
                gpu=gpu,
                model_type=model_type,
                restore_type=denoise_model,
                pretrained_model=custom_model,
                device=device,
            )

        else:
            model = models.CellposeModel(
                gpu=gpu,
                model_type=model_type,
                pretrained_model=custom_model,
                device=device,
            )

        results = model.eval(
            image,
            diameter=nuclei_diam_px,
            channels=channels,
            channel_axis=channel_axis,
            normalize=normalize,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            **kwargs,
        )

        # TODO: unpack below conditionally
        # masks, flows, styles, diams = results if normal
        # masks, flows, styles, diams, img_denoised = results if denoise

        results_dict = {}
        results_dict["masks"] = results[0]
        results_dict["flows"] = results[1]
        results_dict["styles"] = results[2]
        # results_dict["diams"] = results[3] # NOTE: No diams produced?
        if denoise_model:
            results_dict["img_denoised"] = results[3]

        return results_dict

    def segment_selection(
        self,
        scale: str,
        segmentation_channel: str | list[str],
        model_type: CP_DEFAULT_MODELS_typed,
        nuclei_diam_um: float,
        channel_merge_method: Literal["max", "mean", "sum", "median"] = "max",
    ):
        """Perform cell segmentation on a select subset region(s) of the given
        image."""
        raise NotImplementedError()

    def segment_all(
        self,
        scale: str,
        segmentation_channel: str | list[str],
        tiling_shapes: gpd.GeoDataFrame | None,
        model_type: CP_DEFAULT_MODELS_typed,
        nuclei_diam_um: float,
        channel_merge_method: Literal["max", "mean", "sum", "median"] = "max",
        optional_nuclear_channel: str | None = None,
        tiling_shapes_annotation_column: str | None = None,
        normalize: bool = True,
        cellprob_threshold: float = 0.0,
        flow_threshold: float = 0.4,
        custom_model: Path | str | bool = False,
        denoise_model: CP_DENOISE_MODELS_typed | None = None,
        # verbose=True,
        # show_results=False,
        # denoise_model=None,_"
        debug: bool = False,
        preview: bool = False,
        # nuclear_channel: str | None = None,
        **kwargs,
    ) -> None | DataArray:
        """Perform cell segmentation on the given image using cellpose. Writes
        the segmentation results as label elements to the underlying sdata
        object. Also writes a table of the segmentation instances (cells).

        If `tiling_shapes` is specified, then multiple `Labels` are ALSO created
        for every region in `tiling_shapes`.

        Args:
            scale: The scale to segment.
            segmentation_channel: The channel to segment.
            tiling_shapes: Shapes to explicitly tile by. Usually this will be
                the shapes denoting TMA core regions.
            model_type: The type of cellpose model to use.
            nuclei_diam_um: The diameter of the nuclei in microns. If this is
                negative, then the diameter is automatically estimated.
            channel_merge_method: The method to merge multiple channels.
                Options: "max", "mean", "sum", "median".
            optional_nuclear_channel: The optional nuclear channel (if
                supported).
            tiling_shapes_annotation_column: The column in `tiling_shapes` to
                annotate distinct regions by.
            normalize: If True, does image intensity normalization.
            cellprob_threshold: The cell probability threshold.
            flow_threshold: The flow threshold.
            custom_model: The custom model to use, if supplied.
            denoise_model: The denoiser model to use.
            debug: If True, only processes the first two tiles in
                `tiling_shapes`.
            preview: If True, only returns the input image (i.e. everything up
                to just before segmentation).
            **kwargs: Passed to `cellpose.models.*.eval`.

        Returns:
            None if `preview` is False, otherwise the input image.
        """

        # Chosen segmentation scale
        multichannel_image = self.get_image_by_scale(scale=scale)  # CYX

        if isinstance(segmentation_channel, str):
            segmentation_channel = [segmentation_channel]

        # Chosen channel / channels; -> DataArray
        selected_channel_image = multichannel_image.sel(c=segmentation_channel)

        # Log scaling, if not multiscale.
        ds_factor = self.get_downsampling_factor(selected_channel_image)

        # If multiple segmentation channels, merge
        if isinstance(segmentation_channel, list):
            selected_channel_image = self.get_multichannel_image_projection(
                selected_channel_image,  # DataArray
                segmentation_channel,
                method=channel_merge_method,
            )

        input_image = selected_channel_image.transpose("x", "y", "c")
        input_channels_cellpose = [0, 0]  # Grayscale, no nuclear channel
        # If optional nuclear channel is provided;
        if optional_nuclear_channel:
            nuclear_channel_image = multichannel_image.sel(
                c=optional_nuclear_channel
            )
            input_image = xr.concat(
                [selected_channel_image, nuclear_channel_image], dim="c"
            ).transpose("x", "y", "c")
            input_channels_cellpose = [
                1,
                2,
            ]  # Assume nuclear channel is the last channel
        if preview:
            # input_image_ms = to_multiscale(
            #     input_image, DEFAULT_MULTISCALE_DOWNSCALE_FACTORS
            # )
            # return self.get_image_flat(input_image_ms)
            return input_image

        else:
            # Prepare cellpose inputs
            channel_axis = 2 if optional_nuclear_channel else None
            if not optional_nuclear_channel:
                # Recollapse c dim if no nuclear channel ->
                input_image = input_image.squeeze("c")

            # Extract tiles
            # Assume geometries exist in "global" and scale0
            # TODO: Apply `upscale_transformations` to geoms / tiling_shapes
            if tiling_shapes is not None:
                geoms = tiling_shapes["geometry"]

                bboxes = [x.bounds for x in geoms]
                bboxes_rast = [[int(z) for z in x] for x in bboxes]

                if ds_factor != 1:
                    bboxes_rast = [
                        [int(z / ds_factor) for z in x] for x in bboxes_rast
                    ]

                if debug:
                    bboxes_rast = [bboxes_rast[0], bboxes_rast[-1]]
                if (
                    tiling_shapes_annotation_column
                    and tiling_shapes_annotation_column
                    in tiling_shapes.columns
                ):
                    bbox_labels = list(
                        tiling_shapes[tiling_shapes_annotation_column]
                    )
                else:
                    if "tma_label" in tiling_shapes.columns:
                        bbox_labels = list(tiling_shapes["tma_label"])
                    else:  # last resort
                        bbox_labels = list(geoms.index.astype(str))

                # Prepare image tiles
                image_tiles = []
                for bbox in bboxes_rast:
                    xmin, ymin, xmax, ymax = bbox
                    tile = input_image.isel(
                        x=slice(xmin, xmax + 1), y=slice(ymin, ymax + 1)
                    )
                    image_tiles.append(
                        tile.data
                    )  # append the numpy/dask array

                logger.info(
                    f"Segmenting {len(image_tiles)} regions", flush=True
                )
                results = self.cellpose_segmentation(
                    image=image_tiles,
                    model_type=model_type,
                    channels=input_channels_cellpose,
                    channel_axis=channel_axis,
                    nuclei_diam_um=nuclei_diam_um,
                    normalize=normalize,
                    cellprob_threshold=cellprob_threshold,
                    flow_threshold=flow_threshold,
                    custom_model=custom_model,
                    denoise_model=denoise_model,
                    # progress=True,
                    **kwargs,
                )

                # Repack results into global image
                current_max = 0
                label_map = {}
                local_tables = []
                working_shape = (
                    multichannel_image.sizes["x"],
                    multichannel_image.sizes["y"],
                )
                global_seg_mask = np.zeros(working_shape, dtype=np.int32)
                for i, bbox in enumerate(bboxes_rast):
                    ix = i + 1
                    logger.info(
                        f"Processing {bbox_labels[i]} {ix}/{len(bboxes_rast)}",
                        flush=True,
                    )
                    xmin, ymin, xmax, ymax = bbox
                    seg_mask = results["masks"][i].astype(np.int32)
                    local_max = seg_mask.max()
                    seg_mask[seg_mask != 0] += current_max

                    # NOTE: below future implementation
                    # Save as distinct cs's; NOTE: data duplication revisit this
                    # affine_matrix = np.array(
                    #     [
                    #         [1, 0, xmin],
                    #         [0, 1, ymin],
                    #         [0, 0, 1],
                    #     ]
                    # )

                    # subset_translation = Translation(
                    #     [xmin, ymin], axes=("x", "y")
                    # )
                    # bbox_map = Affine(affine_matrix, ("x", "y"), ("x", "y"))
                    # local_transformation_sequence = Sequence(
                    #     transformations + [subset_translation]
                    # )

                    # self.add_label(
                    #     seg_mask,
                    #     self.image_name + f"_labels_{bbox_labels[i]}",
                    #     write_element=True,
                    #     dims=("x", "y"),
                    #     transformations={
                    #         "global": local_transformation_sequence,
                    #         bbox_labels[i]: bbox_map,
                    #         },
                    #     scale_factors=DEFAULT_MULTISCALE_DOWNSCALE_FACTORS,
                    # )
                    LOCAL_CELL_INDEX_LABEL = CELL_INDEX_LABEL + "_local"
                    local_seg_table = pd.DataFrame(
                        index=range(1, 1 + local_max)
                    )
                    local_seg_table = local_seg_table.reset_index(
                        names=LOCAL_CELL_INDEX_LABEL
                    )
                    local_seg_table["tma_label"] = bbox_labels[i]
                    str_index = (
                        local_seg_table[LOCAL_CELL_INDEX_LABEL].astype(str)
                        + "_"
                        + bbox_labels[i]
                    )  # i.e.) '0_A-1', '1_A-1', '2_A-1', ...
                    local_seg_table.index = str_index.values
                    local_seg_table["lyr"] = self.image_name + "_labels"
                    local_tables.append(local_seg_table)

                    # Append to global mask for a one-click all
                    global_seg_mask[xmin : xmax + 1, ymin : ymax + 1] = (
                        seg_mask
                    )

                    new_max = global_seg_mask.max()  # seg_mask.max()
                    if debug and i == 1:
                        i = -1
                    label_map[(current_max + 1, new_max)] = bbox_labels[i]
                    logger.info(f"Current max{current_max}", flush=True)
                    current_max = new_max
                    logger.info(f"New max {current_max}", flush=True)
                    if debug and i == -1:
                        break

                seg_table = pd.concat(local_tables)
                seg_table[CELL_INDEX_LABEL] = range(1, 1 + seg_table.shape[0])

            # Tiling shapes is none -> Single output
            else:
                logger.info(
                    f"Segmenting entirety of {self.image_name}",
                    flush=True,
                )
                results = self.cellpose_segmentation(
                    image=input_image.data,
                    model_type=model_type,
                    channels=input_channels_cellpose,
                    channel_axis=channel_axis,
                    nuclei_diam_um=nuclei_diam_um,
                    normalize=normalize,
                    cellprob_threshold=cellprob_threshold,
                    flow_threshold=flow_threshold,
                    custom_model=custom_model,
                    denoise_model=denoise_model,
                    # progress=True,
                    **kwargs,
                )
                global_seg_mask = results["masks"].astype(np.int32)
            #     seg_table = pd.DataFrame(
            #         index=range(1, 1 + global_seg_mask.max())
            #     )
            #     seg_table = seg_table.reset_index(names=CELL_INDEX_LABEL)
            #     seg_table["tma_label"] = "global"
            #     str_index = seg_table[CELL_INDEX_LABEL].astype(str) + "_global"
            #     seg_table.index = str_index.values
            #     seg_table["lyr"] = self.image_name + "_labels"

            # seg_table["tma_label"] = seg_table["tma_label"].astype("category")
            transformation_sequence = get_transformation(
                selected_channel_image
            )
            return global_seg_mask, transformation_sequence

            # self.add_table(
            #     seg_table,
            #     "labels_expression",
            #     write_element=True,
            #     # TODO: future -> seg_table["tma_label"].unique().tolist()
            #     region=self.image_name + "_labels",  # or none
            #     region_key="lyr",
            #     instance_key=CELL_INDEX_LABEL,
            # )

    def save_segmentation(
        self,
        global_seg_mask: DataArray,
        transformation_sequence: BaseTransformation,
        label_name: str,
        write_element: bool = False,
    ) -> None:
        self.add_label(
            global_seg_mask,
            label_name,
            write_element=write_element,
            dims=("x", "y"),  # TODO: should be reading parent image axes;
            transformations={"global": transformation_sequence},
            # scale_factors=DEFAULT_MULTISCALE_DOWNSCALE_FACTORS, # multiscale
        )


class TMAMeasurer(MultiScaleImageOperations):
    """Class for measuring statistics of cells in TMA cores. Uses skimage.
    measure.regionprops_table for the measurements."""

    EXPORT_PROPERTIES = [
        "label",
        "area",
        "centroid",
        "eccentricity",
        "solidity",
        "intensity_mean",
    ]

    EXTENDED_EXPORT_PROPERTIES = [
        "axis_major_length",
        "axis_minor_length",
        "inertia_tensor",
        "inertia_tensor_eigvals",
        "orientation",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def measure_labels(
        self,
        labels: DataArray,  # Instance Labels
        labels_name: str | None = None,
        tiling_shapes: gpd.GeoDataFrame | None = None,
        extended_properties: bool = False,
        intensity_mode: Literal["mean", "median"] = "mean",
        scale="scale0",
    ) -> None:
        """Measure the properties of the labels in the image. Writes the
        measurements as a table to the underlying sdata object.

        Args:
            labels: The labels to measure. Usually the cell segmentation masks.
            tiling_shapes: The tiling shapes to use.
            extended_properties: If True, includes extended properties.
            intensity_mode: The intensity mode to use. Options: "mean",
                "median".
        """
        intensity_image = self.get_image_by_scale(scale=scale)  # Fullscale

        # Validate that labels has the same x and y dims as intensity_image
        if (
            labels.shape != intensity_image.shape[1:]
        ):  # Exclude channel c dim, yx only
            print(labels.shape)
            print(intensity_image.shape[1:])
            raise ValueError(
                "Labels and intensity image must have the same shape."
            )
        ds_factor = self.get_downsampling_factor(intensity_image)
        properties = self.EXPORT_PROPERTIES
        if extended_properties:
            properties += self.EXTENDED_EXPORT_PROPERTIES

        def intensity_median(mask, intensity_image):
            return np.median(intensity_image[mask], axis=0)

        def _measure_intensities_in_labels(
            labels,
            intensity_image,
            properties,
            intensity_mode,
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
            labels = labels.transpose("x", "y")
            arr = labels.data
            if not isinstance(arr, np.ndarray):
                arr = arr.compute()

            intensity_image = intensity_image.transpose("x", "y", "c")
            if intensity_mode == "mean":
                label_props_table = skimage.measure.regionprops_table(
                    arr,  # DataArray + Dask -> np.array
                    intensity_image=intensity_image.data.compute(),
                    properties=properties,
                )
            elif intensity_mode == "median":
                if "intensity_mean" in properties:
                    properties.remove("intensity_mean")
                label_props_table = skimage.measure.regionprops_table(
                    arr,  # DataArray + Dask -> np.array
                    intensity_image=intensity_image.data.compute(),
                    properties=properties,
                    extra_properties=(intensity_median,),
                )
            else:
                raise ValueError("Unsupported intensity aggregation method.")
            label_props_table = pd.DataFrame(label_props_table)
            # Extract the intensities as expression data
            intensities = label_props_table.filter(like="intensity", axis=1)
            obs_like = label_props_table.drop(columns=intensities.columns)
            print(obs_like)
            print(intensities)
            print(properties)
            return intensities, obs_like

        if tiling_shapes is None:
            intensities, obs_like = _measure_intensities_in_labels(
                labels,
                intensity_image,
                properties,
                intensity_mode,
            )
            intensities.index = intensities.index + 1  # 1-indexed
            obs_like.index = obs_like.index + 1
            intensities.index = intensities.index.astype(str) + "_" + "global"
            obs_like.index = obs_like.index.astype(str) + "_" + "global"
            obs_like["tma_label"] = "global"
            obs_like["tma_label"] = obs_like["tma_label"].astype("category")

        # Tiled regionprops
        else:
            geoms = tiling_shapes["geometry"]
            bboxes = [x.bounds for x in geoms]
            bboxes_rast = [[int(z) for z in x] for x in bboxes]
            if ds_factor != 1:
                bboxes_rast = [
                    [int(z / ds_factor) for z in x] for x in bboxes_rast
                ]

            # Prepare tiles
            intensity_tiles = []
            label_tiles = []

            for bbox in bboxes_rast:
                xmin, ymin, xmax, ymax = bbox
                intensity_tiles.append(
                    intensity_image.isel(
                        x=slice(xmin, xmax + 1), y=slice(ymin, ymax + 1)
                    )
                )
                label_tiles.append(
                    labels.isel(
                        x=slice(xmin, xmax + 1), y=slice(ymin, ymax + 1)
                    )
                )

            tile_labels = tiling_shapes["tma_label"].values

            # TODO: parallelisable? tiles above are list of Dask arrays..
            # NOTE: may not be due to non-seriable functions of regionprops..
            intensity_tables = []
            obs_tables = []
            for i in range(len(bboxes_rast)):
                sub_intensities, sub_obs_like = _measure_intensities_in_labels(
                    label_tiles[i],
                    intensity_tiles[i],
                    properties,
                    intensity_mode,
                )
                sub_obs_like["tma_label"] = tile_labels[i]
                sub_intensities.index = sub_intensities.index + 1
                sub_obs_like.index = sub_obs_like.index + 1
                sub_intensities.index = (
                    sub_intensities.index.astype(str) + "_" + tile_labels[i]
                )
                sub_obs_like.index = (
                    sub_obs_like.index.astype(str) + "_" + tile_labels[i]
                )
                intensity_tables.append(sub_intensities)
                obs_tables.append(sub_obs_like)

            # Merge tables
            intensities = pd.concat(intensity_tables)
            obs_like = pd.concat(obs_tables)

        # Consolidate results
        if labels_name is None:
            labels_name = self.image_name + "_labels"
        obs_like["lyr"] = labels_name
        obs_like["tma_label"] = obs_like["tma_label"].astype("category")
        obs_like = obs_like.rename(columns={"label": CELL_INDEX_LABEL})
        # Extract channel information from the intensity image, assumed to be
        # our dataarray
        channel_names = intensity_image.coords["c"].values
        channel_map = dict(enumerate(channel_names))
        intensities.columns = intensities.columns.map(
            lambda x: channel_map[int(x.split("-")[-1])]
        )  # convert intensity_*-0 to DAPI, etc.

        obs_like = obs_like.rename(
            columns={
                "centroid-0": "centroid_x",
                "centroid-1": "centroid_y",
            }
        )

        new_var = pd.DataFrame(
            index=pd.Series(intensities.columns, name="Marker")
        )
        new_var["intensity_mode"] = intensity_mode

        spatial_px = obs_like[["centroid_x", "centroid_y"]].values
        spatial_um = spatial_px / self.get_px_per_um()
        adata = ad.AnnData(
            intensities.values,
            obs=obs_like,
            obsm={"spatial_px": spatial_px, "spatial_um": spatial_um},
            var=new_var,
        )

        # adata_parsed = TableModel.parse(adata)

        return adata, CELL_INDEX_LABEL

    def save_measurements(
        self,
        adata: AnnData,
        label_name: str,
        output_table_name: str | None = None,
        instance_key: str | None = None,
    ) -> None:
        if output_table_name is None:
            output_table_name = label_name + "_expression"

        if instance_key is None:
            instance_key = CELL_INDEX_LABEL

        self.add_table(
            adata,
            output_table_name,
            write_element=False,
            # TODO: future -> seg_table["tma_label"].unique().tolist()
            region=label_name,  # or none
            region_key="lyr",
            instance_key=instance_key,
        )


def mask_tma(
    spatialdata: SpatialData,
    image_name: str,
    channel: str | list[str],
    scale: str,
    output_mask_name: str,
    sigma_um: float,
    expansion_um: float,
    li_threshold: bool = False,
    edge_filter: bool = False,
    adapt_hist: bool = False,
    gamma_correct: bool = False,
    fill_holes: bool = False,
    remove_small_spots_size: int | float = 0,
    estimated_core_diameter_um: float | int = 700,
    core_fraction: float | int = 0.2,
    rasterize: bool = True,
    reference_coordinate_system: str = "global",
    inplace: bool = True,
) -> SpatialData:
    """
    Masks a channel or channels of the raw multiscale image stored in the .image
    attribute of the provided SpatialData object.

    Args:
        spatialdata: The spatialdata object to mask.
        image_name: The name of the image to mask.
        channel: The channel or channels to mask.
        scale: The scale to mask.
        output_mask_name: The name of the output mask.
        sigma_um: The sigma in microns for the Gaussian filter.
        expansion_um: The expansion in microns for the mask.
        li_threshold: If True, uses the Li thresholding method the image before
            masking.
        edge_filter: If True, applies an edge filter to the image before
            masking.
        adapt_hist: If True, applies adaptive histogram equalization to the
            image before masking.
        gamma_correct: If True, applies gamma correction to the image before
            masking.
        fill_holes: If True, fills holes in the mask.
        remove_small_spots_size: The size in pixels to remove small spots from
            the mask.
        estimated_core_diameter_um: The estimated core diameter in microns.
        core_fraction: The core fraction to use for masking.
        rasterize: If True, rasterizes the mask.
        reference_coordinate_system: The reference coordinate system to use.
        inplace: If True, modifies the spatialdata object in place. If False,
            returns the mask and transformation sequence.

    Returns:
        If inplace is True, returns the modified spatialdata object. If False,
        returns a tuple of the masks, transformation sequence, and masks_gdf.

    """
    # Build model
    model = TMAMasker(
        sdata=spatialdata,
        image_name=image_name,
        reference_coordinate_system=reference_coordinate_system,
    )

    # 4 tuple; masks, transfseq, channel label, masks_gdf
    results = model.mask_tma_cores(
        channel=channel,
        scale=scale,
        sigma_um=sigma_um,
        expansion_um=expansion_um,
        li_threshold=li_threshold,
        edge_filter=edge_filter,
        adapt_hist=adapt_hist,
        gamma_correct=gamma_correct,
        fill_holes=fill_holes,
        remove_small_spots_size=remove_small_spots_size,
        estimated_core_diameter_um=estimated_core_diameter_um,
        core_fraction=core_fraction,
        rasterize=rasterize,
    )

    masks, transformations, _, masks_gdf = results

    if inplace:
        # Add mask to sd
        model.save_masks(
            masks,
            output_mask_name,
            transformations,
            write_element=False,
        )

        # Add shapes/poly representation
        if masks_gdf is not None:
            model.save_shapes(
                masks_gdf,
                output_mask_name,
                transformations,
                write_element=False,
            )
        # Return the modified sdata object;
        return model.sdata
    else:
        # Return the outputs;
        return masks, transformations, masks_gdf


def dearray_tma(
    spatialdata: SpatialData,
    label_name: str,
    expected_diameter_um: float | int = 700,
    expectation_margin: float | int = 0.2,
    expected_rows: int = 0,
    expected_cols: int = 0,
    inplace: bool = True,
) -> (
    SpatialData
    | tuple[GeoDataFrame, GeoDataFrame, AnnData, BaseTransformation]
):
    """
    Automatically dearrays a TMA on masks stored in the .labels attribute of
    the provided spatialdata object.

    Args:
        spatialdata: The spatialdata object to dearray.
        label_name: The name of the label to dearray.
        expected_diameter_um: The expected diameter of the TMA cores in microns.
        expectation_margin: The margin of error for the expected diameter.
        expected_rows: The expected number of rows in the TMA.
        expected_cols: The expected number of columns in the TMA.
        inplace: If True, modifies the spatialdata object in place. If False,
            returns the dearrayed TMA cores and envelope.

    Returns:
        If inplace is True, returns the modified spatialdata object. If False,
        returns a tuple of the dearrayed TMA cores, envelope, and
        transformations.
    """
    # Build model
    model = TMADearrayer(sdata=spatialdata, image_name=label_name)

    # Look for masks_gdf if provided
    masks_gdf = model.sdata.shapes.get(label_name + "_poly", None)

    # Run dearray
    results = model.dearray_and_envelope_tma_cores(
        expected_diameter_um=expected_diameter_um,
        expectation_margin=expectation_margin,
        expected_rows=expected_rows,
        expected_cols=expected_cols,
        masks_gdf=masks_gdf,
    )
    tma_cores, tma_envelope, envelopes, transforms = results

    if inplace:
        model.save_dearray_results(tma_cores, tma_envelope, envelopes)
        return model.sdata
    else:
        return results


def segment_tma(
    spatialdata: SpatialData,
    image_name: str,
    output_segmentation_label: str,
    segmentation_channel: str | list[str],  # Segmentation channel(s)
    tiling_shapes: gpd.GeoDataFrame | str | None = None,
    model_type: Literal[
        "cyto3",
        "cyto2",
        "cyto",
        "nuclei",
        "tissuenet_cp3",
        "livecell_cp3",
        "yeast_PhC_cp3",
        "yeast_BF_cp3",
        "bact_phase_cp3",
        "bact_fluor_cp3",
        "deepbacs_cp3",
        "cyto2_cp3",
    ] = "nuclei",
    nuclei_diam_um: float | None = None,
    channel_merge_method: Literal["max", "mean", "sum", "median"] = "max",
    optional_nuclear_channel: str | None = None,
    tiling_shapes_annotation_column: str | None = None,
    normalize: bool = True,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    custom_model: Path | str | bool = False,
    denoise_model: (
        Literal[
            "nan",
            "denoise_cyto3",
            "deblur_cyto3",
            "upsample_cyto3",
            "oneclick_cyto3",
            "denoise_cyto2",
            "deblur_cyto2",
            "upsample_cyto2",
            "oneclick_cyto2",
            "denoise_nuclei",
            "deblur_nuclei",
            "upsample_nuclei",
            "oneclick_nuclei",
        ]
        | None
    ) = None,
    preview: bool = False,
    reference_coordinate_system: str = "global",
    scale="scale0",
    inplace: bool = True,
    **kwargs,
) -> SpatialData | DataArray | tuple[DataArray, BaseTransformation]:
    """
    Performs cell segmentation using Cellpose to the given image.

    Args:
        spatialdata: The spatialdata object to segment.
        image_name: The name of the image to segment.
        output_segmentation_label: The name of the output segmentation label.
        segmentation_channel: The channel or channels to segment.
        tiling_shapes: Shapes to explicitly tile by. Usually this will be the
            shapes denoting TMA core regions.
        model_type: The type of cellpose model to use.
        nuclei_diam_um: The diameter of the nuclei in microns. If this is
            negative, then the diameter is automatically estimated.
        channel_merge_method: The method to merge multiple channels.
            Options: "max", "mean", "sum", "median".
        optional_nuclear_channel: The optional nuclear channel (if
            supported).
        tiling_shapes_annotation_column: The column in `tiling_shapes` to
            annotate distinct regions by. This is added as a label to each cell.
        normalize: If True, does image intensity normalization.
        cellprob_threshold: The cell probability threshold.
        flow_threshold: The flow threshold.
        custom_model: The custom model to use, if supplied.
        denoise_model: The denoiser model to use.
        preview: If True, only returns the input image (i.e. everything up
            to just before segmentation).
        reference_coordinate_system: The reference coordinate system to use.
        scale: The scale to segment.
        inplace: If True, modifies the spatialdata object in place. If False,
            returns the segmentation mask and transformation sequence.

    Returns:
        If inplace is True, returns the modified spatialdata object. If False,
        returns a tuple of the segmentation mask and transformation sequence.
    """
    # build model
    if isinstance(segmentation_channel, str):
        segmentation_channel = [segmentation_channel]

    model = TMASegmenter(
        sdata=spatialdata,
        image_name=image_name,
        reference_coordinate_system=reference_coordinate_system,
    )

    def _get_shapely_affine_from_matrix(matrix):
        """Get a shapely affine from a matrix."""
        a = matrix[0, 0]
        b = matrix[0, 1]
        d = matrix[1, 0]
        e = matrix[1, 1]

        xoff = matrix[0, 2]
        yoff = matrix[1, 2]

        return (a, b, d, e, xoff, yoff)

    if tiling_shapes is not None and isinstance(tiling_shapes, str):
        ts_transforms = get_transformation_between_coordinate_systems(
            spatialdata,
            spatialdata[tiling_shapes],
            reference_coordinate_system,
        )
        affine = ts_transforms.to_affine_matrix(("x", "y"), ("x", "y"))
        shapely_affine = _get_shapely_affine_from_matrix(affine)
        tiling_shapes = spatialdata.shapes[tiling_shapes].copy()
        tiling_shapes["geometry"] = tiling_shapes["geometry"].affine_transform(
            shapely_affine
        )

    output = model.segment_all(
        scale=scale,
        segmentation_channel=segmentation_channel,
        tiling_shapes=tiling_shapes,
        model_type=model_type,
        nuclei_diam_um=nuclei_diam_um,
        channel_merge_method=channel_merge_method,
        optional_nuclear_channel=optional_nuclear_channel,
        tiling_shapes_annotation_column=tiling_shapes_annotation_column,
        normalize=normalize,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        custom_model=custom_model,
        denoise_model=denoise_model,
        preview=preview,
        **kwargs,
    )
    segmentation_mask, transformations = output

    if inplace:
        model.save_segmentation(
            segmentation_mask, transformations, output_segmentation_label
        )
        return model.sdata
    else:
        return segmentation_mask, transformations


def measure_tma(
    spatialdata: SpatialData,
    image_name: str,
    segmentation_name: str,
    output_table_name: str,
    tiling_shapes: str | None = None,
    extended_properties: bool = False,
    intensity_mode: Literal["mean", "median"] = "mean",
    reference_coordinate_system: str = "global",
    inplace: bool = True,
    scale="scale0",
) -> SpatialData | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Measures the properties of the labels or cells in the image from a given
    labels mask. Writes the measurements as an AnnData table to the underlying
    sdata object.

    Args:
        spatialdata: The spatialdata object to measure.
        image_name: The name of the image to measure.
        segmentation_name: The name of the segmentation mask.
        output_table_name: The name of the output table.
        tiling_shapes: The tiling shapes to use.
        extended_properties: If True, includes extended properties.
        intensity_mode: The intensity mode to use. Options: "mean", "median".
        reference_coordinate_system: The reference coordinate system to use.
        inplace: If True, modifies the spatialdata object in place. If False,
            returns the measurements as a tuple of DataFrames.
        scale: The scale to measure.

    Returns:
        If inplace is True, returns the modified spatialdata object. If False,
        returns a tuple of DataFrames containing the measurements.
    """
    model = TMAMeasurer(
        sdata=spatialdata,
        image_name=image_name,
        reference_coordinate_system=reference_coordinate_system,
    )

    segmentation_data_arr = model.sdata.labels[segmentation_name]

    def _get_shapely_affine_from_matrix(matrix):
        """Get a shapely affine from a matrix."""
        a = matrix[0, 0]
        b = matrix[0, 1]
        d = matrix[1, 0]
        e = matrix[1, 1]

        xoff = matrix[0, 2]
        yoff = matrix[1, 2]

        return (a, b, d, e, xoff, yoff)

    if tiling_shapes is not None and isinstance(tiling_shapes, str):
        ts_transforms = get_transformation_between_coordinate_systems(
            spatialdata,
            spatialdata[tiling_shapes],
            reference_coordinate_system,
        )
        affine = ts_transforms.to_affine_matrix(("x", "y"), ("x", "y"))
        shapely_affine = _get_shapely_affine_from_matrix(affine)
        tiling_shapes = spatialdata.shapes[tiling_shapes].copy()
        tiling_shapes["geometry"] = tiling_shapes["geometry"].affine_transform(
            shapely_affine
        )

    results = model.measure_labels(
        labels=segmentation_data_arr,
        labels_name=segmentation_name,
        tiling_shapes=tiling_shapes,
        extended_properties=extended_properties,
        intensity_mode=intensity_mode,
        scale=scale,
    )

    adata, CELL_INDEX_LABEL = results

    if inplace:
        model.save_measurements(
            adata,
            segmentation_name,
            output_table_name=output_table_name,
            instance_key=CELL_INDEX_LABEL,
        )
        return model.sdata
    else:
        return results


def create_core_sdata(
    sdata: SpatialData,
    tma_table_name: str = "tma_table",
    image_name: str | None = None,
    labels_name: str | None = None,
    mode: Literal["individual", "consolidated"] = "consolidated",
    core_labels: list[str] | None = None,
) -> SpatialData | list[SpatialData]:
    """
    Extract individual TMA cores as separate SpatialData elements or objects.

    This function creates per-core spatial data structures where each core
    has its own local coordinate system starting at (0, 0). Each core contains
    its cropped image, labels, and associated table data.

    Args:
        sdata: SpatialData object containing TMA data.
        tma_table_name: Name of the table containing TMA core metadata.
        image_name: Name of the image to extract cores from. If None, uses the
            first multiscale image found.
        labels_name: Name of the labels to extract. If None, no labels extracted.
        mode: Output mode:
            - "individual": Returns list of SpatialData objects (one per core)
            - "consolidated": Returns single SpatialData with per-core elements
                (e.g., "A1_image", "A1_labels")
        core_labels: List of specific core labels to extract (e.g., ["A-1", "B-2"]).
            If None, extracts all cores from tma_table.

    Returns:
        If mode="individual": List of SpatialData objects, one per core.
        If mode="consolidated": Single SpatialData object with per-core elements.

    """
    # Get TMA core metadata
    if tma_table_name not in sdata.tables:
        raise ValueError(f"Table '{tma_table_name}' not found in SpatialData")

    tma_table = sdata.tables[tma_table_name]

    # Determine which cores to extract
    if core_labels is None:
        if "tma_label" not in tma_table.obs.columns:
            raise ValueError("tma_table must have 'tma_label' column")
        core_labels = tma_table.obs["tma_label"].tolist()

    # Auto-detect image name if not provided
    if image_name is None:
        if len(sdata.images) == 0:
            raise ValueError("No images found in SpatialData")
        image_name = list(sdata.images.keys())[0]

    if image_name not in sdata.images:
        raise ValueError(f"Image '{image_name}' not found in SpatialData")

    # Get the envelope shapes to determine core bounding boxes
    envelope_name = "tma_envelope"
    if envelope_name not in sdata.shapes:
        raise ValueError(
            f"Shape '{envelope_name}' not found. Run TMADearrayer first."
        )

    envelopes_gdf = sdata.shapes[envelope_name]

    # Storage for extracted cores
    if mode == "individual":
        core_sdatas = []
    else:  # consolidated
        consolidated_images = {}
        consolidated_labels = {}
        # consolidated_shapes = {}
        consolidated_tables = {}

    # Process each core
    for core_label in core_labels:
        # Get core envelope
        core_envelope = envelopes_gdf[envelopes_gdf["tma_label"] == core_label]
        if len(core_envelope) == 0:
            logger.warning(
                f"Core {core_label} not found in envelopes, skipping"
            )
            continue

        # Get bounding box
        bounds = core_envelope.geometry.iloc[
            0
        ].bounds  # (minx, miny, maxx, maxy)
        xmin, ymin, xmax, ymax = bounds
        # width = xmax - xmin
        # height = ymax - ymin

        # Extract image crop
        image_data = sdata.images[image_name]
        if isinstance(image_data, DataTree):
            # Multiscale image - use scale0
            image_arr = image_data["scale0"].to_array()
        else:
            image_arr = image_data

        # Crop image (assumes axes: c, y, x)
        core_image = image_arr.sel(
            x=slice(xmin, xmax),
            y=slice(ymin, ymax),
        )

        # Create local coordinate system transformation
        # Translation to move core origin to (0, 0)
        translation_to_local = Translation([-xmin, -ymin], axes=("x", "y"))

        # Get original transformations
        original_transforms = get_transformation(image_arr, get_all=True)

        # Create transformations for core
        core_transforms = {}
        for cs_name, transformations in original_transforms.items():
            # Chain: original transform -> translate to local coords
            core_transforms[cs_name] = Sequence(
                [transformations, translation_to_local]
            )

        # Add local coordinate system (identity)
        core_transforms["local"] = translation_to_local

        # Create core image model
        core_image_model = Image2DModel.parse(
            core_image,
            transformations=core_transforms,
            dims=("c", "y", "x"),
        )

        # Extract labels if provided
        core_labels_model = None
        if labels_name is not None and labels_name in sdata.labels:
            labels_data = sdata.labels[labels_name]

            # Crop labels
            core_labels_arr = labels_data.sel(
                x=slice(xmin, xmax),
                y=slice(ymin, ymax),
            )

            # Create labels model with same transformations
            core_labels_model = Labels2DModel.parse(
                core_labels_arr,
                transformations=core_transforms,
                dims=("y", "x"),
            )

        # Extract relevant table rows
        core_table = None
        if labels_name is not None:
            # Look for table associated with this labels layer
            for table_name, table in sdata.tables.items():
                if table_name == tma_table_name:
                    continue  # Skip the TMA metadata table

                # Check if table is linked to labels
                if (
                    hasattr(table.uns, "get")
                    and "spatialdata_attrs" in table.uns
                ):
                    region = table.uns["spatialdata_attrs"].get("region")
                    if region == labels_name:
                        # Filter table for this core
                        if "tma_label" in table.obs.columns:
                            core_table = table[
                                table.obs["tma_label"] == core_label
                            ].copy()
                            # Update region to point to new labels
                            core_table.uns["spatialdata_attrs"]["region"] = (
                                f"{core_label}_labels"
                                if mode == "consolidated"
                                else "labels"
                            )
                        break

        # Build SpatialData for this core
        if mode == "individual":
            core_sdata_dict = {"images": {"image": core_image_model}}

            if core_labels_model is not None:
                core_sdata_dict["labels"] = {"labels": core_labels_model}

            if core_table is not None:
                core_sdata_dict["tables"] = {"table": core_table}

            core_sdata = SpatialData(**core_sdata_dict)
            core_sdatas.append(core_sdata)

        else:  # consolidated
            consolidated_images[f"{core_label}_image"] = core_image_model

            if core_labels_model is not None:
                consolidated_labels[f"{core_label}_labels"] = core_labels_model

            if core_table is not None:
                consolidated_tables[f"{core_label}_table"] = core_table

    # Return results based on mode
    if mode == "individual":
        return core_sdatas
    else:
        sdata_dict = {}
        if consolidated_images:
            sdata_dict["images"] = consolidated_images
        if consolidated_labels:
            sdata_dict["labels"] = consolidated_labels
        if consolidated_tables:
            sdata_dict["tables"] = consolidated_tables

        return SpatialData(**sdata_dict)


def get_core_metadata(
    sdata: SpatialData,
    tma_table_name: str = "tma_table",
) -> pd.DataFrame:
    """
    Extract metadata DataFrame for TMA cores.

    Args:
        sdata: SpatialData object containing TMA data.
        tma_table_name: Name of the table containing TMA core metadata.

    Returns:
        DataFrame with core metadata indexed by tma_label.
    """
    if tma_table_name not in sdata.tables:
        raise ValueError(f"Table '{tma_table_name}' not found in SpatialData")

    tma_table = sdata.tables[tma_table_name]

    # Convert to DataFrame
    metadata_df = tma_table.obs.copy()

    # Set tma_label as index if it exists
    if "tma_label" in metadata_df.columns:
        metadata_df = metadata_df.set_index("tma_label")

    return metadata_df


def calculate_grid_layout(
    core_labels: list[str],
    metadata: pd.DataFrame | None = None,
    grouping_vars: list[str] | None = None,
    layout: Literal["horizontal", "vertical"] = "horizontal",
    cores_per_row: int | None = None,
) -> dict[str, tuple[int, int]]:
    """
    Calculate grid positions for TMA cores based on metadata grouping.

    Args:
        core_labels: List of core labels to arrange.
        metadata: DataFrame with core metadata indexed by tma_label.
        grouping_vars: List of metadata columns to group by. If None, cores
            are simply sorted by name.
        layout: Layout direction for groups:
            - "horizontal": Groups arranged left-to-right
            - "vertical": Groups arranged top-to-bottom
        cores_per_row: Number of cores per row. If None, auto-calculated based
            on number of cores (approximately square grid).

    Returns:
        Dictionary mapping core_label -> (row, col) grid position.
    """
    # Sort cores
    sorted_cores = sorted(core_labels)

    # Group cores if grouping variables provided
    if grouping_vars is not None and metadata is not None:
        # Verify grouping variables exist
        missing_vars = set(grouping_vars) - set(metadata.columns)
        if missing_vars:
            raise ValueError(
                f"Grouping variables not found in metadata: {missing_vars}"
            )

        # Create grouping key
        groups = {}
        for core_label in sorted_cores:
            if core_label in metadata.index:
                group_key = tuple(
                    metadata.loc[core_label, var] for var in grouping_vars
                )
                if group_key not in groups:
                    groups[group_key] = []
                groups[group_key].append(core_label)
            else:
                # Handle cores without metadata
                if "unknown" not in groups:
                    groups["unknown"] = []
                groups["unknown"].append(core_label)

        # Sort groups by key
        sorted_groups = sorted(groups.items())

        # Arrange cores within each group
        arranged_cores = []
        for _, group_cores in sorted_groups:
            arranged_cores.extend(sorted(group_cores))

        sorted_cores = arranged_cores

    # Calculate grid dimensions
    n_cores = len(sorted_cores)
    if cores_per_row is None:
        # Approximately square grid
        cores_per_row = int(np.ceil(np.sqrt(n_cores)))

    # Assign grid positions
    grid_positions = {}
    for idx, core_label in enumerate(sorted_cores):
        row = idx // cores_per_row
        col = idx % cores_per_row
        grid_positions[core_label] = (row, col)

    return grid_positions


def arrange_cores_in_grid(
    sdata: SpatialData | list[SpatialData],
    tma_table_name: str = "tma_table",
    grouping_vars: list[str] | None = None,
    layout: Literal["horizontal", "vertical"] = "horizontal",
    padding_factor: float = 0.2,
    cores_per_row: int | None = None,
    coordinate_system_name: str | None = None,
) -> SpatialData:
    """
    Arrange TMA cores in a grid layout with a new coordinate system.

    This function creates a new coordinate system that positions cores in a grid
    arrangement. Cores can be organized by metadata variables (e.g., treatment
    groups, responder status).

    Args:
        sdata: Either:
            - Single SpatialData with per-core elements (from create_core_sdata
              with mode="consolidated")
            - List of SpatialData objects (from create_core_sdata with
              mode="individual")
        tma_table_name: Name of the table containing TMA core metadata.
        grouping_vars: List of metadata columns to group cores by (e.g.,
            ["Response"]). Cores with the same group values will be placed
            together. If None, cores are sorted by name.
        layout: Layout direction for groups (currently uses default grid).
        padding_factor: Spacing between cores as fraction of average core
            width/height. Default 0.2 = 20% spacing.
        cores_per_row: Number of cores per row. If None, creates approximately
            square grid.
        coordinate_system_name: Name for the new grid coordinate system. If None,
            auto-generated as "grid_global" or "grid_sorted_by_{vars}".

    Returns:
        SpatialData object with grid coordinate system added to all elements.

    Example:
        >>> # Simple grid arrangement
        >>> sdata_grid = arrange_cores_in_grid(sdata_cores)
        >>>
        >>> # Arrange by response status
        >>> sdata_grid = arrange_cores_in_grid(
        ...     sdata_cores,
        ...     grouping_vars=["Response"],
        ...     padding_factor=0.3
        ... )
        >>> # Cores with Response="Responder" on left, "Non-responder" on right
    """
    # Handle input type
    if isinstance(sdata, list):
        # Convert list of SpatialData to consolidated
        # For now, raise error - need to implement concatenation
        raise NotImplementedError(
            "List input not yet supported. Use mode='consolidated' in create_core_sdata"
        )
    else:
        # Consolidated mode
        consolidated_sdata = sdata

    # Extract core labels from element names
    core_labels = []
    for img_name in consolidated_sdata.images:
        if img_name.endswith("_image"):
            core_label = img_name.replace("_image", "")
            core_labels.append(core_label)

    if len(core_labels) == 0:
        raise ValueError(
            "No cores found in SpatialData (expected elements like 'A-1_image')"
        )

    # Get metadata if grouping requested
    metadata = None
    if (
        grouping_vars is not None
        and tma_table_name in consolidated_sdata.tables
    ):
        metadata = get_core_metadata(consolidated_sdata, tma_table_name)

    # Calculate grid positions
    grid_positions = calculate_grid_layout(
        core_labels=core_labels,
        metadata=metadata,
        grouping_vars=grouping_vars,
        layout=layout,
        cores_per_row=cores_per_row,
    )

    # Determine coordinate system name
    if coordinate_system_name is None:
        if grouping_vars is not None:
            var_str = "_".join(grouping_vars)
            coordinate_system_name = f"grid_sorted_by_{var_str}"
        else:
            coordinate_system_name = "grid_global"

    # Calculate core dimensions for spacing
    # Use first core as reference
    first_core_label = core_labels[0]
    first_core_img = consolidated_sdata.images[f"{first_core_label}_image"]

    # Get image dimensions
    if isinstance(first_core_img, DataTree):
        img_arr = first_core_img["scale0"].to_array()
    else:
        img_arr = first_core_img

    core_height = img_arr.sizes["y"]
    core_width = img_arr.sizes["x"]

    # Calculate spacing
    spacing_y = core_height * (1 + padding_factor)
    spacing_x = core_width * (1 + padding_factor)

    # Apply grid transformations to all elements
    for core_label in core_labels:
        row, col = grid_positions[core_label]

        # Calculate translation to grid position
        grid_y = row * spacing_y
        grid_x = col * spacing_x

        grid_translation = Translation([grid_x, grid_y], axes=("x", "y"))

        # Apply to image
        img_name = f"{core_label}_image"
        if img_name in consolidated_sdata.images:
            img_element = consolidated_sdata.images[img_name]

            # Get existing transformations
            existing_transforms = get_transformation(img_element, get_all=True)

            # Add grid transformation
            # Chain: local coords -> grid coords
            existing_transforms[coordinate_system_name] = Sequence(
                [
                    existing_transforms.get(
                        "local", Translation([0, 0], axes=("x", "y"))
                    ),
                    grid_translation,
                ]
            )

            # Update transformations
            set_transformation(img_element, existing_transforms, set_all=True)

        # Apply to labels
        labels_name = f"{core_label}_labels"
        if labels_name in consolidated_sdata.labels:
            labels_element = consolidated_sdata.labels[labels_name]

            existing_transforms = get_transformation(
                labels_element, get_all=True
            )
            existing_transforms[coordinate_system_name] = Sequence(
                [
                    existing_transforms.get(
                        "local", Translation([0, 0], axes=("x", "y"))
                    ),
                    grid_translation,
                ]
            )
            set_transformation(
                labels_element, existing_transforms, set_all=True
            )

    return consolidated_sdata


def segment_cores_individually(
    sdata: SpatialData,
    image_name: str,
    segmentation_channel: str | list[str],
    tma_table_name: str = "tma_table",
    core_labels: list[str] | None = None,
    model_type: Literal[
        "cyto3",
        "cyto2",
        "cyto",
        "nuclei",
        "tissuenet_cp3",
        "livecell_cp3",
        "yeast_PhC_cp3",
        "yeast_BF_cp3",
        "bact_phase_cp3",
        "bact_fluor_cp3",
        "deepbacs_cp3",
        "cyto2_cp3",
    ] = "nuclei",
    nuclei_diam_um: float | None = None,
    channel_merge_method: Literal["max", "mean", "sum", "median"] = "max",
    optional_nuclear_channel: str | None = None,
    normalize: bool = True,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    custom_model: Path | str | bool = False,
    denoise_model: str | None = None,
    scale: str = "scale0",
    **kwargs,
) -> SpatialData:
    """
    Segment TMA cores individually with per-core label masks.

    This function extracts each core, segments it independently, and creates
    per-core label masks (e.g., "A-1_labels", "A-2_labels"). This approach
    provides better organization for downstream per-core analysis.

    Args:
        sdata: SpatialData object containing TMA data.
        image_name: Name of the image to segment.
        segmentation_channel: Channel(s) to use for segmentation.
        tma_table_name: Name of the table containing TMA core metadata.
        core_labels: List of specific cores to segment. If None, segments all cores.
        model_type: Cellpose model type to use.
        nuclei_diam_um: Expected nucleus diameter in microns. If None, auto-estimated.
        channel_merge_method: Method to merge multiple channels.
        optional_nuclear_channel: Optional nuclear channel for cyto models.
        normalize: Whether to normalize image intensities.
        cellprob_threshold: Cell probability threshold for Cellpose.
        flow_threshold: Flow threshold for Cellpose.
        custom_model: Path to custom Cellpose model.
        denoise_model: Denoising model to use.
        scale: Image scale to use for segmentation.
        **kwargs: Additional arguments passed to Cellpose.

    Returns:
        SpatialData object with per-core label masks added.

    Example:
        >>> sdata_segmented = segment_cores_individually(
        ...     sdata,
        ...     image_name="TMA_image",
        ...     segmentation_channel="DAPI",
        ...     nuclei_diam_um=10.0,
        ... )
        >>> print(sdata_segmented.labels.keys())
        >>> # ['A-1_labels', 'A-2_labels', 'B-1_labels', ...]
    """
    # Extract cores as consolidated SpatialData
    cores_sdata = create_core_sdata(
        sdata=sdata,
        tma_table_name=tma_table_name,
        image_name=image_name,
        labels_name=None,  # No existing labels
        mode="consolidated",
        core_labels=core_labels,
    )

    # Get core labels from images
    if core_labels is None:
        core_labels = [
            name.replace("_image", "")
            for name in cores_sdata.images
            if name.endswith("_image")
        ]

    # Segment each core
    labels_dict = {}

    for core_label in core_labels:
        core_image_name = f"{core_label}_image"

        if core_image_name not in cores_sdata.images:
            logger.warning(f"Core {core_label} not found, skipping")
            continue

        logger.info(f"Segmenting core {core_label}")

        # Create temporary SpatialData for this core
        temp_sdata = SpatialData(
            images={image_name: cores_sdata.images[core_image_name]}
        )

        # Segment using existing function
        try:
            segmented_sdata = segment_tma(
                spatialdata=temp_sdata,
                image_name=image_name,
                output_segmentation_label="labels",
                segmentation_channel=segmentation_channel,
                tiling_shapes=None,  # No tiling - already cropped
                model_type=model_type,
                nuclei_diam_um=nuclei_diam_um,
                channel_merge_method=channel_merge_method,
                optional_nuclear_channel=optional_nuclear_channel,
                normalize=normalize,
                cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold,
                custom_model=custom_model,
                denoise_model=denoise_model,
                scale=scale,
                inplace=True,
                **kwargs,
            )

            # Extract the labels
            labels_dict[f"{core_label}_labels"] = segmented_sdata.labels[
                "labels"
            ]

        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to segment core {core_label}: {e}")
            continue

    # Add labels to the cores SpatialData
    for label_name, label_data in labels_dict.items():
        cores_sdata.labels[label_name] = label_data

    return cores_sdata


def measure_cores_individually(
    sdata: SpatialData,
    image_name: str,
    tma_table_name: str = "tma_table",
    core_labels: list[str] | None = None,
    extended_properties: bool = False,
    intensity_mode: Literal["mean", "median"] = "mean",
    scale: str = "scale0",
) -> SpatialData:
    """
    Measure properties of cells in per-core label masks.

    This function measures cell properties for each core's label mask and
    creates per-core measurement tables.

    Args:
        sdata: SpatialData object with per-core images and labels.
        image_name: Original image name (used to find core images like "A-1_image").
        tma_table_name: Name of the table containing TMA core metadata.
        core_labels: List of specific cores to measure. If None, measures all cores.
        extended_properties: Whether to compute extended morphological properties.
        intensity_mode: How to aggregate intensity values ("mean" or "median").
        scale: Image scale to use for measurements.

    Returns:
        SpatialData object with per-core measurement tables added.

    Example:
        >>> sdata_measured = measure_cores_individually(
        ...     sdata_segmented,
        ...     image_name="TMA_image",
        ...     extended_properties=True,
        ... )
        >>> print(sdata_measured.tables.keys())
        >>> # ['A-1_table', 'A-2_table', 'B-1_table', ...]
    """
    # Get core labels from label elements
    if core_labels is None:
        core_labels = [
            name.replace("_labels", "")
            for name in sdata.labels
            if name.endswith("_labels")
        ]

    # Measure each core
    for core_label in core_labels:
        core_image_name = f"{core_label}_image"
        core_labels_name = f"{core_label}_labels"

        if core_image_name not in sdata.images:
            logger.warning(f"Core image {core_image_name} not found, skipping")
            continue

        if core_labels_name not in sdata.labels:
            logger.warning(
                f"Core labels {core_labels_name} not found, skipping"
            )
            continue

        logger.info(f"Measuring core {core_label}")

        # Create temporary SpatialData for this core
        temp_sdata = SpatialData(
            images={image_name: sdata.images[core_image_name]},
            labels={core_labels_name: sdata.labels[core_labels_name]},
        )

        # Measure using existing function
        try:
            measured_sdata = measure_tma(
                spatialdata=temp_sdata,
                image_name=image_name,
                segmentation_name=core_labels_name,
                output_table_name=f"{core_label}_table",
                tiling_shapes=None,  # No tiling
                extended_properties=extended_properties,
                intensity_mode=intensity_mode,
                scale=scale,
                inplace=True,
            )

            # Extract and add the table
            if f"{core_label}_table" in measured_sdata.tables:
                core_table = measured_sdata.tables[f"{core_label}_table"]
                # Add core label to table metadata
                core_table.obs["tma_label"] = core_label
                sdata.tables[f"{core_label}_table"] = core_table

        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to measure core {core_label}: {e}")
            continue

    return sdata
