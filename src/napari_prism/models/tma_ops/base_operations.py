from threading import Lock
from typing import Any, Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from dask.array import Array
from geopandas import GeoDataFrame
from loguru import logger
from numpy import dtype, float64, ndarray
from shapely import Point, Polygon
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
    get_transformation_between_coordinate_systems,
)
from xarray import DataArray, DataTree

__all__ = [
    "SdataImageOperations",
    "SingleScaleImageOperations",
    "MultiScaleImageOperations",
]


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
            label = label.data.compute()

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
