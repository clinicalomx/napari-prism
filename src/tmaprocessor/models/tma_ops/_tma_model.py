import torch
from pathlib import Path
import xarray as xr
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import shapely
import geopandas
import skimage
from shapely import Polygon, Point
from math import pi
from functools import partial
import math
import pandas as pd
import geopandas as gpd
from skimage import feature, transform
from shapely import Polygon, geometry
from itertools import product
from cellpose import core, models, io, denoise
import tifffile
import gc
from uuid import uuid4
import anndata as ad
import os
import numpy as np
from sklearn.cluster import KMeans
from abc import ABC, abstractmethod
from typing import List, Literal, Union, Any, Tuple, Callable
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel, ShapesModel, TableModel
from spatialdata.models.models import Schema_t
from spatialdata.transformations import \
    get_transformation_between_coordinate_systems, BaseTransformation, Scale, \
    Translation, Sequence
from geopandas import GeoDataFrame
from datatree.datatree import DataTree # Image pyramid class
from xarray import DataArray
from numpy import ndarray, float64, dtype
from dask.array import Array
from anndata import AnnData
# For cellpose api
from napari.qt.threading import thread_worker
try:
    from torch import no_grad
except ImportError:
    def no_grad():
        def _deco(func):
            return func
        return _deco

__all__ = ["TMAMasker", "TMADearrayer", "TMASegmenter", "TMAMeasurer"]

class SdataImageOperations():
    MULTICHANNEL_PROJECTION_METHODS = ["max", "mean", "sum", "median"]
    def __init__(
        self, 
        sdata: SpatialData, 
        image_name: str, 
        reference_coordinate_system: str = "global"
    ) -> None:
        self.sdata: SpatialData = sdata
        self.image_name: str = image_name
        self.reference_coordinate_system: str = reference_coordinate_system

    def get_image(self):
        """ Get the image for masking given an index in the image pyramid to
            retrieve a certain resolution, as well as the provided channel. """
        raise NotImplementedError("calling from abstract method")

    def apply_image_correction(
            self,
            image: DataArray | ndarray[Any, dtype[float64]],
            contrast_limits: Tuple[float, float],
            gamma: float = 1.0,
    ) -> DataArray | ndarray[Any, dtype[float64]]:
        """ Apply contrast limits and gamma correction to an image. 
        https://napari.org/stable/_modules/napari/layers/image/image.html#Image
        NOTE: consider this a module func """
        con_min, con_max = contrast_limits
        image = (image - con_min) / (con_max - con_min)
        
        if isinstance(image, DataArray):
            image = image.clip(0, con_max)
        else:
            np.clip(image, 0, con_max, out=image)
        image = image ** gamma
        return image

    def get_transformation_to_cs(self, cs: str) -> BaseTransformation:
        """ Convert current image to the chosen coordinate system. """
        if cs not in self.sdata.coordinate_systems:
            raise ValueError(f"No {cs} coordinate system in sdata") 
        
        return get_transformation_between_coordinate_systems(
            self.sdata,
            self.get_image(),
            cs)
    
    def get_unit_scaling_factor(self, coordinate_system: str) -> BaseTransformation:
        """ Get the scaling factor to convert pixel (global) units to the chosen 
            coordinate system. 
            
            Temporary implementation, following below for updates:
            - https://github.com/scverse/spatialdata/issues/30 
            - https://github.com/scverse/spatialdata/issues/436

        """
        #base_transformation = self.get_transformation_to_cs(coordinate_system)
        base_transformation = get_transformation_between_coordinate_systems(
            self.sdata,
            "global", # Px
            coordinate_system
        )
        scaling_factor_x = base_transformation.to_affine_matrix("x", "x")[0, 0]
        scaling_factor_y = base_transformation.to_affine_matrix("y", "y")[0, 0]
        assert scaling_factor_x == scaling_factor_y, "Unequal scaling factors for X and Y"
        return scaling_factor_x
    
    def get_px_per_um(self):
        # Check if um is in the sdata
        if "um" not in self.sdata.coordinate_systems:
            raise ValueError("No um coordinate system in sdata")

        return 1 / self.get_unit_scaling_factor(coordinate_system="um") # px per micron

    def convert_um_to_px(self, um_value: int | float) -> int:
        """ # um > px """
        # round to nearest whole number since this is px; 
        # this is where rasterization occurs, so rounding errors happen here
        px_value = np.round(um_value * self.get_px_per_um())
        return int(px_value)

    def get_multichannel_image_projection(
        self,
        image: DataArray,
        channels: List[str],
        method: Literal["max", "mean", "sum", "median"] = "max",
    ) -> DataArray:
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
        c_name = f'{method}[{"_".join(channels)}]'# + "_" + method
        compacted = compacted.assign_coords(c=c_name)
        return compacted
    # def create_cs_from_shapes(
    #     self,
    #     shapes_element_name: str, # Name of the shapes element in sdata
    #     coordinate_system_column: str, # column in shapes element to use as cs
    #     coordinate_system_name: str | None = None, # optional Suffix to add
    # ) -> None:
    #     """
    #     Create a new coordinate system in contained SpatialData object, based on
    #     a contained shapes element.
    #     """
    #     assert shapes_element_name in self.sdata, \
    #         f"{shapes_element_name} not in sdata"
        
    #     shapes = self.sdata[shapes_element_name]

    #     if not isinstance(shapes, GeoDataFrame):
    #         raise ValueError("Shapes element is not a GeoDataFrame")
        
    #     if coordinate_system_column not in shapes.columns:
    #         raise ValueError(f"{coordinate_system_column} not in shapes")
        
    #     if coordinate_system_name is None:
    #         coordinate_system_name = ""

    #     # Create a new coordinate system based on the unique values in the column

    # Overwrite element; -> Refresh widgets containing relevant layers    
    def overwrite_element(
        self,
        sdata: SpatialData,
        element: Schema_t,
        element_name: str
    ) -> None:
        
        def _delete_from_disk(
            sdata: SpatialData, 
            element_name: str, 
            overwrite: bool
        ) -> None:
            if element_name in sdata \
                and len(sdata.locate_element(sdata[element_name])) != 0:
                if overwrite:
                    del sdata[element_name]
                    sdata.delete_element_from_disk(element_name)
                else:
                    raise OSError(
                        f"`{element_name}` already exists. Use overwrite=True to rewrite.")
                
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
        **kwargs
    ) -> None:
        """ Adds single or multiscale image to contained SpatialData. 
            Arguments passed to Image2DModel.parse.
            """
        image = Image2DModel.parse(image, *args, **kwargs)
        #new_image_name = f"{self.image_name}_{image_suffix}"
        if write_element:
            self.overwrite_element(self.sdata, image, image_label)
        else:
            self.sdata[image_label] = image
            logger.warning("Spatialdata object is not stored on disk, could only add element in memory.")

    def add_label(
        self, 
        label: ndarray[Any, dtype[float64]] | DataArray | Array,
        label_name: str,
        write_element: bool = False,
        *args, 
        **kwargs
    ) -> None:
        """ Adds single or multiscale image to contained SpatialData. 
            Arguments passed to Image2DModel.parse.

            Mimick _write_element_to_disk func from nap-sd._viewer.SpatialDataViewer
            """
        # Cast to dask -> DataArrays dont liike inplace brush erasing / inpainting
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
        #new_label_name = f"{self.image_name}_{label_suffix}"
        if write_element:
            self.overwrite_element(self.sdata, label, label_name)
        else:
            self.sdata[label_name] = label
            logger.warning("Spatialdata object is not stored on disk, could only add element in memory.")
    
    def add_shapes(
        self, 
        shapes: GeoDataFrame,
        shapes_name: str | None = None,
        write_element: bool = False,
        parent_table: TableModel | None = None,
        *args, 
        **kwargs
    ) -> None:
        shapes = ShapesModel.parse(shapes, *args, **kwargs)
        # if shapes_suffix is not None:
        #     shapes_suffix = f"{self.image_name}_{shapes_suffix}"
        # else:
        #     shapes_suffix = f"{self.image_name}"
        if shapes_name is None:
            shapes_name = "shapes"
        
        if write_element:
            self.overwrite_element(self.sdata, shapes, shapes_name)
        else:
            self.sdata[shapes_name] = shapes
            logger.warning("Spatialdata object is not stored on disk, could only add element in memory.")

    def _get_scaled_polygon(self, polygon, scale) -> Polygon:
        """ Returns the polygons, but re-scaled to 'real' world measurements. """
        scaled_coords = [(x*scale, y*scale) for x,y in polygon.exterior.coords]
        return Polygon(scaled_coords)
    
    def _get_scaled_point(self, point, scale) -> Point:
        """ Returns the circular polygons, but re-scaled to 'real' world 
            measurements. """
        return Point(point.x * scale, point.y * scale)

    def add_table(
        self, 
        table: AnnData | pd.DataFrame,
        table_suffix: str,
        write_element: bool = False,
        *args, 
        **kwargs
    ) -> None:
        """ Adds single or multiscale image to contained SpatialData. 
            Arguments passed to Image2DModel.parse.

            Mimick _write_element_to_disk func from nap-sd._viewer.SpatialDataViewer
            """
        if isinstance(table, pd.DataFrame):
            table = AnnData(obs=table)

        # Cast to dask -> DataArrays dont liike inplace brush erasing / inpainting
        table = TableModel.parse(table, *args, **kwargs)
        new_table_name = f"{self.image_name}_{table_suffix}"
        
        if write_element:
            self.overwrite_element(self.sdata, table, new_table_name)
        else:
            self.sdata[new_table_name] = table
            logger.warning("Spatialdata object is not stored on disk, could only add element in memory.")

class SingleScaleImageOperations(SdataImageOperations):
    """ DataArrays """
    def __init__(
        self, 
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

    def get_image(self) -> DataArray:
        """ Get the image for masking given an index in the image pyramid to
            retrieve a certain resolution, as well as the provided channel. """
        #ssi = self.sdata.image[self.image_name]
        ssi = self.sdata[self.image_name] # Might be a .labels
        if not isinstance(ssi, DataArray):
            raise ValueError("Image is not a single scale image")
        return ssi
    
    def get_image_channels(self) -> ndarray:
        return self.get_image().coords["c"].data
    
class MultiScaleImageOperations(SingleScaleImageOperations):
    """ DataTrees, containing DataArrays """
    def __init__(
        self, 
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
    
    def get_image(self) -> DataTree:
        """ Get the image pyramid/DataTree. Checks done here to ensure retrieved
            image is of the right type (Multiscale DataTree)"""
        msi = self.sdata.images[self.image_name]
        if not isinstance(msi, DataTree):
            raise ValueError("Image is not a multiscale image")
        return msi
    
    def get_image_channels(self) -> ndarray:
        return self.get_image_by_scale().coords["c"].data

    def get_image_scales(self) -> List[str]:
        msi = self.get_image()
        return [x.lstrip("/") for x in msi.groups[1:]]
    
    def get_image_shapes(self) -> List[str]:
        scales = self.get_image_scales()
        dataarrays = [self.get_image()[x] for x in scales]
        shapes = [x.image.shape for x in dataarrays]
        return shapes

    def get_image_by_scale(self, scale: str | None = None) -> DataArray:
        """ Get the image for masking given an index in the image pyramid to
            retrieve a certain resolution, as well as the provided channel. """
        if scale is None:
            scale = self.get_image_scales()[-1] # smallest resolution

        return self.get_image()[scale].image

    def get_downsampling_factor(
            self, 
            working_image: DataArray | ndarray[Any, dtype[float64]]) -> float:
        """ Get the downsampling factor which convert the 
            working_image to the full scale image. Assumes equal scaling for Y 
            and X between scales, since format is pyramid. """
        fs = self.get_image_by_scale("scale0") # C,Y,X
        fs_x_shape = fs.coords["x"].shape[0]
        fs_y_shape = fs.coords["y"].shape[0]
        if isinstance(working_image, DataArray):
            ds_x_shape = working_image.coords["x"].shape[0]
            ds_y_shape = working_image.coords["y"].shape[0]
        else:
            ds_y_shape, ds_x_shape = working_image.shape
        ds_factor_x = fs_x_shape / ds_x_shape
        ds_factor_y = fs_y_shape / ds_y_shape
        assert ds_factor_x == ds_factor_y, "Unequal downsampling factors for X and Y"
        return ds_factor_x

class TMAMasker(MultiScaleImageOperations):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def mask_dataarray_image(
        self, 
        image: DataArray | DataTree, # For explicit coord
        xmin: int,
        ymin: int,
        xmax: int,
        ymax: int
    ) -> DataArray:
        """ Mask a DataArray image to have values only within the bounding box 
            points. Values outside the bounding box are set to 0. """
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
    ) -> ndarray[dtype[bool]]:
        """ Generates simple 'segmentation' of TMA cores using piped
            operations of a (optional preprocessing) > gaussian blur > 
            multiotsu > threshold > expansion. 
        """
        # Save parameters;
        # For compat; work with Dask
        # data loss shouldnt be an issue since we're working with images of 8-16bit?
        image = image.astype("int64")
        image = image.data.compute()
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
            blur_thresholded, expansion_px)

        return blur_thresholded_expanded

    def generate_mask_contours(
        self, 
        masks: ndarray[dtype[bool]], 
        area_threshold: int | float = 0
    ) -> List[shapely.geometry.Polygon]:
        """ Convert segmentation masks into contour formats (list of vertices),
            then filter contours by a pixel area threshold. 

            Marching swquare method; not hull
        """
        contours = skimage.measure.find_contours(
            masks.astype(np.int32), 0.5) # Bool array; 
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
        core_fraction: int | float = 1
    ) -> float:
        """ Estimate a threshold for including whole or fractional circular 
            objects of a given diameter in px. """
        estimated_core_radius_px = estimated_core_diameter_px / 2

        # Take into account mask expansion 
        estimated_core_radius_px += expansion_px
        estimated_core_area_px2 = pi * (estimated_core_radius_px)**2
        return estimated_core_area_px2 * core_fraction
    
    def generate_mask_bounding_boxes(
        self, 
        polygons: List[shapely.geometry.Polygon], 
        bbox_pad: int = 0
    ) -> List[shapely.geometry.Polygon]:
        """ Returns the unique/separated bounding boxes of a list of shapes 
            (shapely.Polygon), with some padding. 
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
        channel: str,
        scale: str,
        mask_selection: Tuple[int, int, int, int] | None = None, #xmin, ymin, xmax, ymax  -> Masks a selection of the image
        sigma_um: float | int = 10,
        expansion_um: float | int = 10,
        li_threshold: bool = False,
        edge_filter: bool = False,
        adapt_hist: bool = False,
        gamma_correct: bool = False,
        estimated_core_diameter_um: float | int = 700,
        core_fraction: float | int  = 0.2,
        rasterize: bool = True,
        contrast_limits: Tuple[float, float] = None,
        gamma: float = None
    ) -> None:
        """ Main masker function, 

            1) Generate an initial mask using a gaussian blur
            2) Thresholds regions to only be above the area or fraction of an
                area above the estimated area (calculated from user-set core
                diameter estimate in mm)
            3) Generates unique bounding boxes which do not overlap. Overlapping
                regions get one merged bounding box.
            4) Results are cached to the underlying sdata object.  

        """
        # Log transformations to return to global 
        transformations = []

        # Retrieve image to process
        multichannel_image = self.get_image_by_scale(scale) # C, Y, X

        # if isinstance(channel, List):
        #     # Collapse multichannel image to agglomerate channel using method; max
        #     selected_channel_image = self.get_multichannel_image_projection(
        #         multichannel_image, 
        #         channel, 
        #         method=multichannel_projection_method)
        # else:
        #     selected_channel_image = multichannel_image.sel(c=channel) # Y, X

        selected_channel_image = multichannel_image.sel(c=channel)

        # Apply image correction if supplied;
        if (contrast_limits is not None) and (gamma is not None):
            logger.info("Applying image correction")
            logger.info(f"contrast_limits : {contrast_limits}")
            logger.info(f"gamma : {gamma}")

            selected_channel_image = self.apply_image_correction(
                selected_channel_image, 
                contrast_limits, 
                gamma)

        ds_factor = self.get_downsampling_factor(selected_channel_image)
        
        # Upscale transformations
        upscale_transformations = Scale(
            [ds_factor, ds_factor],
            axes=("x", "y"))
        transformations.append(upscale_transformations)

        # Subset to given bounding box --> Need to remap 
        if mask_selection is not None:
            xmin, ymin, xmax, ymax = mask_selection

            # Add translation to the transformations in the original space
            translation_transformation = Translation(
                [xmin, ymin],
                axes=("x", "y"))
            transformations.append(translation_transformation)

            # Remap bounding box to downsampled space
            xmin /= ds_factor
            ymin /= ds_factor
            xmax /= ds_factor
            ymax /= ds_factor
            xmin, ymin = map(math.floor, (xmin, ymin))
            xmax, ymax = map(math.ceil, (xmax, ymax))
            selected_channel_image = selected_channel_image.isel(
                x=slice(xmin, xmax), y=slice(ymin, ymax))
            
        transformation_sequence = Sequence(transformations)

        # convert um parameters to px equivalents; 
        sigma_px = self.convert_um_to_px(sigma_um)
        expansion_px = self.convert_um_to_px(expansion_um)
        estimated_core_diameter_px = self.convert_um_to_px(
            estimated_core_diameter_um)
        
        # account for downsampling
        sigma_px /= ds_factor
        expansion_px /= ds_factor
        estimated_core_diameter_px /= ds_factor

        initial_masks = self.generate_blurred_masks(
            selected_channel_image, 
            sigma_px=sigma_px, 
            expansion_px=expansion_px, 
            li_threshold=li_threshold,
            edge_filter=edge_filter, 
            adapt_hist=adapt_hist, 
            gamma_correct=gamma_correct)
        
        # TODO: below is stilll quite slow, likely to do with transformation
        channel_label = channel if isinstance(channel, str) else "_".join(channel)
        self.add_label(
            initial_masks, 
            f"{channel_label}_mask", 
            write_element=True,
            dims=("y", "x"),
            transformations={"global": transformation_sequence})

        if rasterize:
            self._rasterize_tma_masks(
                initial_masks,
                estimated_core_diameter_px,
                expansion_px,
                core_fraction,
                channel_label,
                transformation_sequence)
        else:
            return transformation_sequence, ds_factor
        
    def _rasterize_tma_masks(
        self,
        initial_masks: ndarray[dtype[bool]],
        estimated_core_diameter_px: int | float,
        expansion_px: int | float,
        core_fraction: int | float,
        channel_label: str,
        transformation_sequence: BaseTransformation
    ) -> None:
        area_threshold = self.estimate_threshold_from_radius(
            estimated_core_diameter_px=estimated_core_diameter_px,
            expansion_px=expansion_px,
            core_fraction=core_fraction)

        masks_polygons = self.generate_mask_contours(
            masks=initial_masks, 
            area_threshold=area_threshold)

        masks_bboxes = self.generate_mask_bounding_boxes(
            polygons=masks_polygons)

        masks_gdf = self.consolidate_geometrical_objects(
            masks_polygons, 
            masks_bboxes)
        
        #masks_table = TableModel.from_geodataframe(masks_gdf)

        self.add_shapes(
            masks_gdf,
            f"{channel_label}_mask_poly",
            write_element=True,
            transformations={"global": transformation_sequence}
        )
        
    def consolidate_geometrical_objects(
        self, 
        masks_polygons: List[shapely.geometry.Polygon],
        masks_bboxes: List[shapely.geometry.Polygon]
    ) -> geopandas.GeoDataFrame:
        """ Consolidate the masks, polygons and bounding boxes into a single
            GeoDataFrame. """
        #downsampling_factor = self.get_downsampling_factor(working_image)
        # scaling_func = partial(
        #     self._get_scaled_polygon, scale=1)
        
        masks_polygons_gs = geopandas.GeoSeries(masks_polygons)

        masks_bboxes_gs = geopandas.GeoSeries(masks_bboxes)

        agg = geopandas.GeoDataFrame(
            {"geometry": masks_polygons_gs, "masks_bboxes": masks_bboxes_gs}, 
            geometry="geometry")
        
        # Reduce poly
        agg["geometry"] = agg["geometry"].simplify(tolerance=2)

        return agg

class TMADearrayer(SingleScaleImageOperations): # Treating TMAMasker as a Mixin, since Dearrayer depends on the Masker
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid_positions = None
        self.grid_labels = None
        self.core_gdf = None
        self.envelope_gdf = None
        self.merged = None # The final TMA relational df

    def estimate_rotation(
        self, 
        blobs_dog: ndarray[Any, dtype[float64]], 
        expected_diameter_px: int | float
    ) -> float:
        """ Estimates the rotation in the image from a list of blobs 
            generated from skimage blobs dog detection. 
            
            Does this in 2 steps:
            1. Detect blobs which are in the same row by checking if they are 
                separated by atleast 2x the expected diameter in the x-axis, 
                but are within the expected diameter in the y-axis. Store all
                these angles across the entire image.
            2. Estimate the angle of rotation of rows by taking the median of 
                the stored angles.
            """
        angles = []
        for blobs in blobs_dog:
            y, x, _ = blobs

            for blobs_next in blobs_dog:
                y_next, x_next, _ = blobs_next

                if (
                    (x_next > x) 
                    and (x_next - x) > expected_diameter_px*2 
                    and abs(y_next - y) < expected_diameter_px
                ):
                    angle = (180/math.pi) * math.atan2(y_next - y, x_next - x)
                    angles.append(angle)
        
        if len(angles) == 0:
            return 0

        return np.median(angles)

    def detect_blobs(
        self, 
        image: ndarray[Any, dtype[float64]] | DataArray | Array,
        expected_diameter_um: int | float,
        expectation_margin: int | float
    ) -> Tuple[ndarray[Any, dtype[float64]], float, ndarray[Any, dtype[float64]]]:
        """ Detect circular blobs of a given radii within some expectation
            margin. Expectation margin is essentially the min and max sigma 
            values for DoG detection. """
        expected_radius_um = expected_diameter_um / 2
        expected_radius_px = self.convert_um_to_px(expected_radius_um)
        logger.info(f"estimated_core_radius_um : {expected_radius_um}")
        logger.info(f"estimated_core_radius_px : {expected_radius_px}")

        # Account for the image's transform;
        decomposed = self.get_image().transform["global"].transformations
        scales = [x for x in decomposed if isinstance(x, Scale)]
        if scales != []: # or len 0
            scale = scales[0].scale[0]
            logger.info(f"working with image with downsampling factor : {scale}")
            expected_radius_px /= scale
            logger.info(f"adjusted expected_radius_px : {expected_radius_px}")

        # Accounts for incorrect diameter estimates
        min_sigma = expected_radius_px * (1 - expectation_margin)
        max_sigma = expected_radius_px * (1 + expectation_margin)

        # Blob detection with blob_dog
        blobs_dog = feature.blob_dog(
            image, 
            min_sigma=min_sigma, 
            max_sigma=max_sigma, 
            threshold=0.1,
            overlap=0.5)
        blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)

        # Rotation correction
        angle = self.estimate_rotation(blobs_dog, expected_radius_px*2)
        logger.info(f"estimated rotation angle : {angle}")
        # Extract centroid for shapely affine rotations as well
        rows, cols = image.shape[0], image.shape[1]
        image_centroid = np.array((rows, cols)) / 2. - 0.5

        blurred_image_rotated = transform.rotate(
            image, 
            angle=angle, 
            resize=False,
            center=image_centroid) # As above, just so its in (rows, cols) format
        
        # Repeat;
        blobs_dog = feature.blob_dog(
            blurred_image_rotated, 
            min_sigma=min_sigma, 
            max_sigma=max_sigma, 
            threshold=0.1,
            overlap=0.5)
        blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)  # Correct radii for DoG
        return blobs_dog, angle, image_centroid # Also return the metadata for processing
        
    def estimate_grid(
        self, 
        blobs_dog: ndarray[Any, dtype[float64]],
        nrows: int = 0, 
        ncols: int = 0
    ) -> Tuple[ndarray[Any, dtype[float64]], ndarray[Any, dtype[float64]]]:
        """ 
            Simple grid estimation algoriithm.

            Estimate and create a grid layout model from the position of the
            blobs in blobs_dog. Uses the following:

            1) A traversal algorithm to segment the points into rows and columns.
            

            2) A Kmeans algorithm to segment the points into an expected number
            of rows and columns. Uses the kmeans algorithm to segment the points 
            into rows and columns.

            From the expected row and column coordiantes, constructs an ideal /
            perfect grid, and the coordinates of each grid point.
        """

        def _walk_and_segment_points(sorted_points, radius):
            """ Estimate the number of 'distinct' intervals or segments in 
                sorted_points, by walking through each point, taking into
                account an estimated radius separating each point. 
                
                Should take in suggested xs and ys.
                """
            previous_point = 0
            row_list = []
            current_list = []
            for i, p in enumerate(sorted_points):
                if i == 0: # First point
                    previous_point = p
                    row_list.append(p)
                elif i == len(sorted_points) - 1: # Last point (most bottom / right)
                    row_mean = np.mean(row_list)
                    current_list.append(row_mean)
                else:
                    if p - previous_point < radius: # If the position of the pt is within a radius dist,
                        row_list.append(p)
                    else:
                        row_mean = np.mean(row_list) # The next pt indicates there's a new row/col
                        current_list.append(row_mean) # So take what we have and use that as the avg row/col pos
                        row_list = []
                        row_list.append(p)
                    previous_point = p
            return current_list

        def _label_elements(N, M):
            """ Label each element in the grid according to a standard TMA
                core formatting.
                
                Rows: Capital Alphabet [A-Z],
                Columns: Numerics [0-9]* 
                Formatting: [A-Z]-[0-9]*
            """
            # Initialize an empty array for the labels
            labels = np.empty((N, M), dtype='object')
            
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
        # Arrange the combinations into a matrix of shape (len(x), len(y), 2); last dim is x,y coord
        grid_positions = np.array(
            [
                combinations[i * len(current_y_list):(i + 1) * len(current_y_list)] 
                for i in range(len(current_x_list))
            ]
        )
        grid_positions = grid_positions.transpose(0, 1, 2)
        grid_labels = _label_elements(len(current_x_list), len(current_y_list))
        return grid_positions, grid_labels

    def estimate_grid_tps(
        self, 
        blobs_dog: ndarray[Any, dtype[float64]],
        nrows: int = 0, 
        ncols: int = 0
    ) -> Tuple[ndarray[Any, dtype[float64]], ndarray[Any, dtype[float64]]]:
        """

        """
        pass

    def dearray(
        self, 
        #image: ndarray[Any, dtype[float64]] | DataArray | Array,
        expected_diameter_um: int | float, 
        expectation_margin: int | float = 0.2,
        expected_rows: int = 0,
        expected_cols: int = 0):
        """ Resolves the grid layout from the initial detection and grid
            estimation results using structured data formats (GeoDataFrames)"""
        image = self.get_image() # Allow for multiple images;
        self.transforms = image.transform # Inherit transforms from the image
        image = image.astype(bool) # enforce binary
        # convert um to px
        #expected_diameter_px = self.convert_um_to_px(expected_diameter_um)

        cores_circles, image_rotation, image_centroid = self.detect_blobs(
            image,
            expected_diameter_um,
            expectation_margin)
        
        #image_rotation *= -1

        grid_positions, grid_labels = self.estimate_grid(
            cores_circles,
            expected_rows,
            expected_cols)

        self.grid_positions = grid_positions
        self.grid_labels = grid_labels
        
        # Create structured data for above
        cores_gdf = gpd.GeoDataFrame(pd.DataFrame(cores_circles))
        cores_gdf.columns = ["y", "x", "radius"]
        cores_gdf["circles"] = (
            cores_gdf
            .apply(lambda x: Point(x["x"], x["y"]) 
            .buffer(x["radius"]), axis=1)
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
                grid_point = Point(grid_positions[i, j][::-1])  # Get the current grid point to test 

                # Get the closest circle -> THis can be erronous;
                closest_idx = cores_gdf.distance(grid_point).idxmin()
                closest = cores_gdf.loc[closest_idx]["circles"]
                # Check if the circle centroid is radius away from the grid point
                circle_found = closest.distance(grid_point) <= radii_threshold
                
                if circle_found:
                    # Check if observed circle has been assigned already
                    assigned_label = cores_gdf.loc[closest_idx, 'tma_label']
                    if assigned_label != MISSING_LABEL:
                        p_i, p_j = np.where(grid_labels == assigned_label)
                        p_i, p_j = p_i[0], p_j[0]
                        previous_point = Point(grid_positions[p_i, p_j][::-1])
                        # Check which is closer
                        if closest.distance(previous_point) < closest.distance(grid_point):
                            # make this missing
                            final_grid_labels[i, j] = MISSING_LABEL
                            continue
                        else:
                            # Make the previous missing
                            final_grid_labels[p_i, p_j] = MISSING_LABEL

                    # Assign the closest circle to that label
                    cores_gdf.loc[closest_idx, 'tma_label'] = grid_labels[i, j]
                    cores_gdf.loc[closest_idx, 'tma_perfect_position_y'] = grid_positions[i, j][0]
                    cores_gdf.loc[closest_idx, 'tma_perfect_position_x'] = grid_positions[i, j][1]
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
            predicate="intersects")
        intersections = joined_gdf[
            joined_gdf['index'] != joined_gdf['index_right']]

        if len(intersections) > 0:
            print("Intersecting cores found, going with 0.95 of diameter")
            return self.dearray(
                expected_diameter_um * 0.95,
                expectation_margin,
                expected_rows,
                expected_cols
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
        cores_gdf["circles"] = cores_gdf["circles"] \
            .rotate(angle=image_rotation, origin=Point(image_centroid))

        # Cache bounding box of each circle
        cores_gdf["circles_bboxes"] = (
            cores_gdf["circles"]
                .map(lambda x: geometry.box(
                    x.bounds[0], x.bounds[1], x.bounds[2], x.bounds[3])
                    )
        )

        # Duplicate representation of points -> TODO: double check if needed or not
        cores_gdf["point"] = (
            cores_gdf[["x", "y"]]
                .apply(lambda x: Point(*x), axis=1)
        )
        cores_gdf = cores_gdf.set_geometry("point") # technically our core representation are Points -> Buffered to be 'circles'
        cores_gdf["point"] = (
            cores_gdf["point"]
                .rotate(angle=image_rotation, origin=Point(image_centroid))
        )
        cores_gdf["x"] = cores_gdf["point"].map(lambda c: c.x)
        cores_gdf["y"] = cores_gdf["point"].map(lambda c: c.y)
        cores_gdf = cores_gdf.set_geometry("circles")

        return cores_gdf

    def _generate_enveloping_bounding_boxes(self, core_gdf, masks_gdf=None):
        """ Merges the bounding boxes of the TMA masks (data transfer from
            core_gdf, provided by controller/viewer/presenter class) and the 
            TMA core to get an all-enveloping bounding box. 
            
            If masks_gdf is None, then just uses the cores.

            If provided, then gets the envelope of the mask + core.
            """
        image = self.get_image()

        scaling_func = partial(
            self._get_scaled_polygon, scale=1)
        
        # Scale up polygon attributes
        core_gdf["circles"] = core_gdf["circles"].map(lambda p: scaling_func(p))
        core_gdf["circles_bboxes"] = (
            core_gdf["circles_bboxes"]
                .map(lambda p: scaling_func(p))
        )
        core_gdf["point"] = (
            core_gdf["point"]
                .map(lambda p: self._get_scaled_point(p, 1))
        )

        # # Scale up scalar attributes ? 
        # core_gdf["x"] = core_gdf["x"] * 1
        # core_gdf["y"] = core_gdf["y"] * 1
        # core_gdf["radius"] = core_gdf["radius"] * 1
        core_gdf.set_geometry('circles_bboxes', inplace=True)

        # Merge super and current geopandasdfs 
        if masks_gdf is None:
            masks_gdf = core_gdf
        
        # Join based on intersecting bounding boxes;
        masks_gdf.set_geometry('masks_bboxes', inplace=True)
        
        intersections = masks_gdf.sjoin(
            core_gdf, # Will have duplicates. core:mask -> 1:N
            how='left', 
            predicate='intersects')
        merged = intersections.merge(
            core_gdf.reset_index()[['index', 'circles_bboxes']], 
            left_on="index_right", right_on="index", how="left")
        # Dictionary to hold the merged geometries for each unique circle_bbox
        merged_geometries = {}
    
        # Iterate over unique circle_bboxes, excluding None values
        for circle_bbox in merged['circles_bboxes'].dropna().unique():
            # Filter rows that intersect with the current circle_bbox
            intersecting_rows = merged[
                (
                    merged['masks_bboxes']
                    .apply(
                        lambda x: x is not None and x.intersects(circle_bbox)
                        )
                )
            ]

            # If there are any intersections, merge them
            if not intersecting_rows.empty:
                all_geometries = (
                    intersecting_rows['masks_bboxes']
                        .tolist()
                )
                all_geometries += [circle_bbox]
                unified_geometry = gpd.GeoSeries(all_geometries).unary_union
                new_box = geometry.box(*unified_geometry.bounds)
                merged_geometries[circle_bbox.wkt] = new_box
            else:
                # If no intersections, use the circle_bbox itself
                merged_geometries[circle_bbox.wkt] = circle_bbox
    
        # Map the merged geometries back to the original DataFrame
        # Handle None values explicitly
        merged['merged_geometry'] = (
            merged['circles_bboxes']
                .apply(
                    lambda x: merged_geometries[x.wkt] 
                        if x is not None else None
                    )
        )

        #TEMP for debuging
        #self.merged = merged 
        # Now we have a set of polygon relationally consistent 'database' of polygons
        # By default the TMA masks without an assigned TMA core have been dropped in the spatial joins
        # But, clean up final outputs
        # Overwrite some objects in sdata / viewer rep
        tma_masks = (
            merged[["geometry", "masks_bboxes", "tma_label"]]
                .rename(columns={"tma_masks":"geometry"}) # duplicate tmas due to multiple masks in one core
        )
        tma_masks = tma_masks.set_geometry("geometry")
        self.masks_bboxes = tma_masks["masks_bboxes"]

        tma_cores = (
            core_gdf[["point", "x", "y", "radius", "circles_bboxes", "tma_label"]]
                .rename(columns={"point":"geometry"})
        )
        tma_cores = tma_cores.set_geometry("geometry")


        self.core_gdf = tma_cores
        self.add_shapes(
            tma_cores,
            "tma_core",
            write_element=True,
            transformations=self.transforms
        )

        tma_envelope = (
            merged[["merged_geometry", "tma_label"]]
                .rename(columns={"merged_geometry":"geometry"}) # The enveloping box which encompasses the mask AND core bboxes. useful for seg tiles
        )
        tma_envelope = tma_envelope.dropna().set_geometry("geometry")
        tma_envelope = tma_envelope.drop_duplicates("tma_label")
        tma_envelope = tma_envelope.reset_index(drop=True)
        self.envelope_gdf = tma_envelope #TODO: resolve cores, but no masks.
        self.add_shapes(
            tma_envelope,
            "tma_envelope",
            write_element=True,
            transformations=self.transforms
        )
        # NOTE: Transformations helps to track the original spaces; but the drawback is inplace writings;
        # transforms = self.transforms["global"].transformations
        # scale = transforms[0].to_affine_matrix("x", "x")[0][0]
        # y_translation, x_translation = transforms[1].translation

        # Add Labels version
        # from rasterio.features import rasterize
        # shapes = [(g, i+1) for i, g in enumerate(tma_envelope["geometry"])]  # Label starting from 1
        # mask = rasterize(shapes, dtype=np.int32, out_shape=self.get_image().shape)
        # self.add_label(
        #     mask,
        #     "tma_envelope_label",
        #     write_element=True,
        #     transformations=self.transforms
        # )

        #TODO: add Table that annotates these objects.
        cores = AnnData(obs=pd.DataFrame(tma_cores["tma_label"]))
        #cores.obs["image"] = self.image_name
        cores.obs["lyr"] = "tma_core"

        envelopes = AnnData(obs=pd.DataFrame(tma_envelope["tma_label"]))
        #envelopes.obs["image"] = self.image_name
        envelopes.obs["tma_label"] = envelopes.obs["tma_label"].astype("str") 
        envelopes.obs["instance_id"] = tma_envelope.index
        envelopes.obs["lyr"] = "tma_envelope" # region; have it match the shapes element name above
        envelopes.obs["lyr"] = envelopes.obs["lyr"].astype("category")
        envelopes.uns["grid_labels"] = self.grid_labels
        envelopes.uns["grid_positions"] = self.grid_positions

        #cores.obs["lyr_lbl"] = self.image_name + "_tma_core"
        # TODO: Need to drop the geoms; these are added as shapesmodel above

        # self.add_table(
        #     cores,
        #     "tma_core_tbl",
        #     write_element=True,
        #     region_key="lyr",
        #     region=self.image_name + "_tma_core",
        #     instance_key="tma_label"
        # )

        self.add_table(
            envelopes,
            "tma_table",
            write_element=True,
            region="tma_envelope", # element name
            region_key="lyr", # region
            instance_key="instance_id" 
        )
        
        # tma_label = pd.DataFrame(tma_envelope["tma_label"])
        # tma_label["index"] = tma_label.index + 1
        # envelope_label = AnnData(obs=tma_label)
        # #envelope_label.obs["image"] = self.image_name
        # envelope_label.obs["lyr"] = self.image_name + "_tma_envelope_label"
        # self.add_table(
        #     envelope_label,
        #     "tma_envelope_label_tbl",
        #     write_element=True,
        #     region_key="lyr",
        #     region=self.image_name + "_tma_envelope_label",
        #     instance_key="index"
        # )

    def dearray_and_envelope_tma_cores(
            self,
            expected_diameter_um,
            expectation_margin=0.2,
            expected_rows=0,
            expected_cols=0,
            masks_gdf=None):
        """ Main dearray function, 
        
            1) Perform the dearrayer.
            2) Merge and resolve intersections between the TMA masks and TMA
                cores to get a envelope. Useful for getting full image subsets
                of each core automatically. """
        core_gdf = self.dearray(
            expected_diameter_um, 
            expectation_margin, 
            expected_rows, 
            expected_cols)
        self._generate_enveloping_bounding_boxes(core_gdf, masks_gdf)

    # TODO: Below operate on the grid, user interaction for manual grid modifications
    def append_tma_row(self):
        grid_labels = self.grid_labels
        nrows, ncols = grid_labels.shape 
        new_row_label = chr(65 + nrows) # py 0 indexing, but shape is count -> so the index of new row will be this
        new_row = [{new_row_label}-{j + 1} for j in range(ncols)]
        grid_labels = np.vstack([grid_labels, new_row])
        self.grid_labels = grid_labels

    def remove_tma_row(self):
        pass

    # Below operate on the relational df
    def _check_tma_df_exists(self):
        if self.merged is None:
            raise AttributeError("Run initial dearray first.") # TODO: the user should be able to add first anway,... but then grid layout etc gets messed
    
    def add_tma_core(self, geometry):
        self._check_tma_df_exists()
        # add tma cores to model

    def remove_tma_core(self):
        self._check_tma_df_exists()
        # remove core entry from list

# TODO: how to access mask information in common sdata; once sdatas more mature
# NOTE: for now access data from the layers
# TODO: Can wrap cellpose-napari, but need to changei mage layer and channel to segment
class TMASegmenter(MultiScaleImageOperations):
    #TEMP_DIR = "./_segmenter_temp"
    
    CP_DEFAULT_MODELS = [
        "cyto3", "cyto2", "cyto", "nuclei", "tissuenet_cp3", "livecell_cp3", 
        "yeast_PhC_cp3", "yeast_BF_cp3", "bact_phase_cp3", "bact_fluor_cp3", 
        "deepbacs_cp3", "cyto2_cp3"]
    
    CP_DENOISE_MODELS = [
        "nan",
        "denoise_cyto3", "deblur_cyto3", "upsample_cyto3", 
        "oneclick_cyto3", "denoise_cyto2", "deblur_cyto2",
        "upsample_cyto2", "oneclick_cyto2", "denoise_nuclei",
        "deblur_nuclei", "upsample_nuclei", "oneclick_nuclei"]

    CP_DEFAULT_MODELS_typed = Literal[
        "cyto3", "cyto2", "cyto", "nuclei", "tissuenet_cp3", "livecell_cp3", 
        "yeast_PhC_cp3", "yeast_BF_cp3", "bact_phase_cp3", "bact_fluor_cp3", 
        "deepbacs_cp3", "cyto2_cp3"]
    
    CP_DENOISE_MODELS_typed = Literal[
        "nan",
        "denoise_cyto3", "deblur_cyto3", "upsample_cyto3", 
        "oneclick_cyto3", "denoise_cyto2", "deblur_cyto2",
        "upsample_cyto2", "oneclick_cyto2", "denoise_nuclei",
        "deblur_nuclei", "upsample_nuclei", "oneclick_nuclei"]

    def __init__(
        self,
        *args, # Assume mulyiscale image 
        **kwargs):
        super().__init__(*args, **kwargs)
        self._gpu_arr = False
        # self.edge_sobel = edge_sobel
        # self.blur_sigma = blur_sigma
        # self.blur_expansion = blur_expansion
        # self.histogram_clip = histogram_clip
        # self.expansion_um = expansion_um
        # self.nuclei_diameter_um = nuclei_diameter_um

        # self.full_image_cell_mask = np.empty(self.image.shape, dtype=np.int32)
        # self.full_image_nuclear_mask = np.empty(self.image.shape, dtype=np.int32)
        # self.full_image_processed = np.empty(self.image.shape, dtype=np.int32)

        # self.tmp_expr_path = f"{self.TEMP_DIR}/{self.image_name}_expr.csv"
        # self.tmp_meta_path = f"{self.TEMP_DIR}/{self.image_name}_meta.csv"

    def setup_segmentation_output_images(self, image_shape):
        self.full_image_cell_mask = np.empty(image_shape, dtype=np.int32)
        self.full_image_nuclear_mask = np.empty(image_shape, dtype=np.int32)
        self.full_image_processed = np.empty(image_shape, dtype=np.int32)

    def preprocess_subimage(
            self, 
            subimg,
            edge_sobel,
            blur_sigma,
            blur_expansion,
            histogram_clip):
        if edge_sobel:
            subimg = skimage.filters.sobel(subimg)
        if blur_sigma > 0:
            blur = skimage.filters.gaussian(subimg, sigma=blur_sigma)
            bts = skimage.filters.threshold_multiotsu(blur)[0]
            blur_thresholded = blur >= bts
        else:
            blur_thresholded = subimg
        # Expand masks
        if blur_expansion > 0:
            blur_thresholded_expanded = skimage.segmentation.expand_labels(
                blur_thresholded, 
                blur_expansion)
        else:
            blur_thresholded_expanded = None
        # Remask on gaussian blur to highlight salient edges;
        if blur_thresholded_expanded is not None:
            subimg = subimg * blur_thresholded_expanded

        if histogram_clip > 0:
            subimg = skimage.exposure.equalize_adapthist(
                subimg, 
                clip_limit=histogram_clip)

        return subimg

    @no_grad()
    def cellpose_segmentation(
        self, 
        image: ndarray[Any, dtype[float64]] | List[ndarray[Any, dtype[float64]]],
        model_type: CP_DEFAULT_MODELS_typed,
        channels: List[int] | List[List[int]], # First elemnt to segment, second optional nuclear -> Handled by widget
        channel_axis: int = 2, # Assume last axis in ndarrays
        nuclei_diam_um: float | None = None, # If none, automated 
        normalize: bool = True, # If True, does image intensity normalization.
        cellprob_threshold: float = 0.0,
        flow_threshold: float = 0.4,
        custom_model: Path | str | bool | None = False,
        denoise_model: CP_DENOISE_MODELS_typed | None = None,
        **kwargs):
        device = None
        # sentinel values for diameter
        if nuclei_diam_um <= 0: 
            nuclei_diam_px = None # Automated diameter estimation
        else:
            nuclei_diam_px = self.convert_um_to_px(nuclei_diam_um) #  * self.get_px_per_um() # px 

        gpu = core.use_gpu()

        # Check if macos + mps
        if not gpu and torch.backends.mps.is_available():
            gpu = True
            device = "mps"
            logger.info("Using MPS")

        # TODO: refine these checks
        if custom_model is None:
            custom_model = False

        if denoise_model == "nan":
            denoise_model = None

        if denoise_model is None:
            denoise_model = False

        if custom_model is None:
            model_type = None

        if denoise_model:
            model = denoise.CellposeDenoiseModel(
                gpu=gpu, 
                model_type=model_type, 
                restore_type=denoise_model,
                pretrained_model=custom_model,
                device=device)
        
        else:        
            model = models.CellposeModel(
                gpu=gpu, 
                model_type=model_type,
                pretrained_model=custom_model,
                device=device)
        
        results = model.eval(
            image, 
            diameter=nuclei_diam_px, 
            channels=channels,
            channel_axis=channel_axis,
            normalize=normalize,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            **kwargs)
        
        #TODO: unpack below conditionally
        # masks, flows, styles, diams = results if normal
        #masks, flows, styles, diams, img_denoised = results if denoise
        
        results_dict = {}
        results_dict["masks"] = results[0]
        results_dict["flows"] = results[1]
        results_dict["styles"] = results[2]
        #results_dict["diams"] = results[3] # NOTE: No diams produced?
        if denoise_model:
            results_dict["img_denoised"] = results[3]
        
        return results_dict
    
    def expand_segmentation_masks(self, masks, expansion_um):
        expansion_px = expansion_um * self.pixels_per_micron # um * px/um = px
        expanded_mask = skimage.segmentation.expand_labels(masks, distance=expansion_px)
        return expanded_mask
    
    def get_bbox_selection(self, bounding_box_index, polygon_col):
        bounding_box = self.gff.loc[bounding_box_index][polygon_col]
        return [int(x) for x in bounding_box.bounds]
    
    def segment_bbox_selection(
            self, 
            image,
            bounding_box_index, 
            polygon_col, 
            nuclei_diam_um,
            expansion_um,
            show_results=False, 
            denoise_model=None,
            **preprocess_kwargs):
        xmin, ymin, xmax, ymax = self.get_bbox_selection(bounding_box_index, polygon_col)
        tma_label = self.gff.loc[bounding_box_index]["tma_label"]
        subimg = image[xmin:xmax+1, ymin:ymax+1] # .transpose("y","x").chunk("auto").values
        adapted_masked_image = self.preprocess_subimage(subimg, **preprocess_kwargs)
        
        # programmatically tile based on gpu
        tile = True
        if core.use_gpu():
            try:
                import torch
                gpu_free_vram = torch.cuda.mem_get_info()[0] / 1024 ** 3
                if gpu_free_vram > 20: # High 20gib threshold;
                    tile = False
            except ImportError:
                pass
        
        # Create a mask for nuclear and expanded nuclear (estimated cell)
        nuclear_masks = self.cellpose_segmentation(
            adapted_masked_image, 
            tma_label, 
            output_dir=None,#self.TEMP_DIR, 
            nuclei_diam_um=nuclei_diam_um, 
            denoise_model=denoise_model,
            tile=tile) # technically do a 'tile' with TMA subsets.
        
        # Can clear assigned memory here;
        if core.use_gpu():
            try:
                with torch.no_grad():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        expanded_masks = self.expand_segmentation_masks(
            nuclear_masks, 
            expansion_um)

        if show_results:
            fig, ax = plt.subplots(ncols=4, figsize=(14,6))
            images = [subimg, adapted_masked_image, skimage.color.label2rgb(nuclear_masks), skimage.color.label2rgb(expanded_masks)]
            titles = ["Original image", "Processed image", "Nuclear masks", "Cell (expanded nuclear) masks"]
            for i, img in enumerate(images):
                if "image" in titles[i]:
                    cmap = "gray"
                else:
                    cmap = "viridis"
                
                ax[i].imshow(img, cmap=cmap)
                ax[i].set_title(titles[i])
                ax[i].axis("off")
            plt.show()

        return adapted_masked_image, nuclear_masks, expanded_masks, xmin, ymin, xmax, ymax

    def get_tma_iter(self, polygon_col="geometry"):
        return self.gff[~self.gff[polygon_col].isna()].drop_duplicates("tma_label")
    
    def get_n_tmas(self, polygon_col="geometry"):
        return len(self.get_tma_iter(polygon_col))

    # Master function;
    def segment_all(
        self, 
        scale: str,
        segmentation_channel: str | List[str],
        tiling_shapes: gpd.GeoDataFrame, # Assume global cs + scale0
        model_type: CP_DEFAULT_MODELS_typed,
        nuclei_diam_um: float,
        channel_merge_method: Literal["max", "mean", "sum", "median"] = "max", # If multiple channels, merges them according to this.
        optional_nuclear_channel: str | None = None,
        tiling_shapes_annotation_column: str | None = None, # Column in tiling shapes for labels. If none, then tiles assigned index
        normalize: bool = True,
        cellprob_threshold: float = 0.0,
        flow_threshold: float = 0.4,
        custom_model: Path | str | bool = False,
        denoise_model: CP_DENOISE_MODELS_typed | None = None,
        # verbose=True, 
        # show_results=False, 
        # denoise_model=None,
        debug: bool = True,
        # nuclear_channel: str | None = None,
        **kwargs):
        # Log transformations to return to global
        transformations = []
    
        # Chosen segmentation scale
        multichannel_image = self.get_image_by_scale(scale) # CYX
        
        # Chosen channel / channels; -> DataArray
        selected_channel_image = multichannel_image.sel(c=segmentation_channel)

        # Log scaling, if not multiscale.
        ds_factor = self.get_downsampling_factor(selected_channel_image)
        upscale_transformations = Scale(
            [ds_factor, ds_factor],
            axes=("x", "y")
        )
        transformations.append(upscale_transformations)

        # If multiple segmentation channels, merge
        if isinstance(segmentation_channel, list):
            selected_channel_image = self.get_multichannel_image_projection(
                selected_channel_image, # DataArray
                segmentation_channel,
                method=channel_merge_method
            )
        
        input_image = selected_channel_image.transpose("x", "y")
        input_channels_cellpose = [0, 0] # Grayscale, no nuclear channel
        # If optional nuclear channel is provided;
        if optional_nuclear_channel:
            nuclear_channel_image = (
                multichannel_image
                    .sel(c=optional_nuclear_channel)
            )
            input_image = xr.concat(
                [selected_channel_image, nuclear_channel_image],
                dim="c"
            ).transpose("x", "y", "c")
            input_channels_cellpose = [1, 2] # Assume nuclear channel is the last channel
        
        # Extract tiles
        # Assume geometries exist in "global" and scale0
        geoms = tiling_shapes["geometry"]
        bboxes = [x.bounds for x in geoms]
        bboxes_rast = [[int(z) for z in x] for x in bboxes]
        if debug:
            bboxes_rast = [bboxes_rast[0], bboxes_rast[-1]]
        if tiling_shapes_annotation_column and \
            tiling_shapes_annotation_column in tiling_shapes.columns:
                bbox_labels = list(tiling_shapes[tiling_shapes_annotation_column])
        else:
            bbox_labels = list(geoms.index)

        # Prepare image tiles
        image_tiles = []
        for bbox in bboxes_rast:
            xmin, ymin, xmax, ymax = bbox
            tile = input_image.isel(
                x=slice(xmin, xmax+1),
                y=slice(ymin, ymax+1)
            )
            image_tiles.append(tile.data) # append the numpy/dask array

        # Prepare cellpose inputs
        channel_axis = 2 if optional_nuclear_channel else None
        results = self.cellpose_segmentation(
            image=image_tiles, #TODO: might be out of ordering;
            model_type=model_type,
            channels=input_channels_cellpose,
            channel_axis=channel_axis,
            nuclei_diam_um=nuclei_diam_um,
            normalize=normalize,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            custom_model=custom_model,
            denoise_model=denoise_model,
            **kwargs
        )

        # Unpack results -> numpy ecosystem
        # Options; 1) Full size image, 2) Repack images into TMA region coord systems
        working_shape = (
            multichannel_image.sizes["x"], 
            multichannel_image.sizes["y"])
        global_seg_mask = np.empty(working_shape, dtype=np.int32)
        
        # Repack results into global image
        # TODO: seg + added
        current_max = 0
        label_map = {}

        for i, bbox in enumerate(bboxes_rast):
            logger.info(f"Processing bbox {i+1}/{len(bboxes_rast)}", flush=True)
            xmin, ymin, xmax, ymax = bbox
            seg_mask = results["masks"][i]
            seg_mask[seg_mask != 0] += current_max
            global_seg_mask[xmin:xmax+1, ymin:ymax+1] = seg_mask

            new_max = seg_mask.max()
            if debug and i == 1:
                i = -1
            label_map[(current_max+1, new_max)] = bbox_labels[i]
            current_max = new_max
            if debug and i == -1:
                break
        
        # Add;
        transformation_sequence = Sequence(transformations)
        self.add_label(
            global_seg_mask,
            self.image_name + "_labels",
            write_element=True,
            dims=("x", "y"),
            transformations={"global": transformation_sequence}
        )

        # TODO: some errors/bugs below.., not all boxes are being segmented or
        # assigned.. 
        interval_index = pd.IntervalIndex.from_tuples(
            list(label_map.keys()), closed="both")
        label = list(label_map.values())
        seg_table = pd.DataFrame(index=range(1, 1 + global_seg_mask.max()))
        seg_table["region_label"] = pd.cut(
            seg_table.index, 
            bins=interval_index, 
            labels=label, 
            include_lowest=True)
        interval_to_label = {
            interval: label for interval, label in 
            zip(interval_index, label_map.values())
            }
        seg_table["region_label"] = seg_table["region_label"].map(
            interval_to_label)
        
        seg_table["lyr"] = self.image_name + "_labels"
        seg_table = seg_table.reset_index()
        self.add_table(
            seg_table,
            f"labels_tbl",
            write_element=True,
            region=self.image_name + "_labels",
            region_key="lyr",
            instance_key="index"
        )

    def export_segmentation_masks(self, imname=None):
        if imname is None:
            imname = self.image_name
        tifffile.imwrite(self.TEMP_DIR + f"/{imname}_nuclear_masks.tif", self.full_image_nuclear_mask, dtype=np.int32)
        tifffile.imwrite(self.TEMP_DIR + f"/{imname}_cell_masks.tif", self.full_image_cell_mask, dtype=np.int32)
    
    def export_expression_table(self, mode="mean", imname=None, full_image=None):
        if imname is None:
            imname = self.image_name

        if full_image is None: # If a reference to the full ndim image isnt provided, load from model
            full_image_ref = self.get_image()[0]#self.image_layer.data[0] # fullscale -> 0; (c, x, y)
            #full_image = full_image.transpose("y", "x", "c").chunk("auto").values
        
        # Try gpu accelerated regionprops with cucim
        try:
            # import cucim.skimage as skimage
            # import cupy as cp
            # import cudf 
            # self.xp = cp
            # self._gpu_arr = True
            raise ImportError()

        except ImportError:
            print("cucim not installed, falling back to skimage", flush=True)
            import skimage
            self.xp = np

        def intensity_median(mask, intensity_image):
            return self.xp.median(intensity_image[mask], axis=0)

        def measure_core_expression_tables(i):
            xmin, ymin, xmax, ymax = self.get_bbox_selection(i, "geometry")
            tma_label = self.gff.loc[i]["tma_label"]
            sub_intensity_image = full_image_ref[:, xmin:xmax+1, ymin:ymax+1]
            #sub_intensity_image = sub_intensity_image.chunk(sub_intensity_image.shape).values # rechunk to subset im; should be comfy in memory
            nuclear_masks = self.full_image_nuclear_mask[xmin:xmax+1, ymin:ymax+1]
            cell_masks = self.full_image_cell_mask[xmin:xmax+1, ymin:ymax+1]

            properties = self.EXPORT_PROPERTIES.copy()

            sub_intensity_image = sub_intensity_image.transpose(1, 2, 0) # Channel in last axis

            sub_intensity_image = self.xp.array(sub_intensity_image) # Casting to whatever backend 
            nuclear_masks = self.xp.array(nuclear_masks)
            cell_masks = self.xp.array(cell_masks)


            if mode == "median":
                properties.remove("intensity_mean")
                nuclear_props_table = skimage.measure.regionprops_table(
                    nuclear_masks,
                    intensity_image=sub_intensity_image,
                    properties=properties,
                    extra_properties=(intensity_median,)
                )

                cell_props_table = skimage.measure.regionprops_table(
                    cell_masks,
                    intensity_image=sub_intensity_image,
                    properties=properties,
                    extra_properties=(intensity_median,)
                )

            elif mode == "mean":
                nuclear_props_table = skimage.measure.regionprops_table(
                    nuclear_masks,
                    intensity_image=sub_intensity_image, 
                    properties=properties,
                )

                cell_props_table = skimage.measure.regionprops_table(
                    cell_masks,
                    intensity_image=sub_intensity_image, 
                    properties=properties,
                )

            else:
                raise ValueError("Unsupported intensity aggregation method.")

            # Gc collect
            del sub_intensity_image
            del nuclear_masks
            del cell_masks
            gc.collect()

            coord_map = {
                "centroid-0": "centroid_x_local",
                "centroid-1": "centroid_y_local",
            }
            # Adjust bbox coordinates
            # Get from gpu memory;
            if self._gpu_arr:
                cpt = cudf.DataFrame(cell_props_table)
                npt = cudf.DataFrame(nuclear_props_table)
            else:
                cpt = pd.DataFrame(cell_props_table)
                npt = pd.DataFrame(nuclear_props_table)

            cpt = cpt.rename(columns=coord_map)
            cpt["centroid_x_global"] = cpt["centroid_x_local"] + xmin
            cpt["centroid_y_global"] = cpt["centroid_y_local"] + ymin

            npt = npt.rename(columns=coord_map)
            npt["centroid_x_global"] = npt["centroid_x_local"] + xmin
            npt["centroid_y_global"] = npt["centroid_y_local"] + ymin
            
            # Add tma information
            cpt["tma_label"] = tma_label
            npt["tma_label"] = tma_label

            return npt, cpt

        num_tmas = self.gff["tma_label"].nunique()
        print(f"found num tmas: {num_tmas}")
        # Memory intensive, so parallelisation might be an issue
        results = [
            measure_core_expression_tables(i) for i in range(num_tmas) #range(1)#
        ]

         # Unpack results and consolidate
        nuc_tables = [result[0] for result in results]
        cell_tables = [result[1] for result in results]
        nuc_all_tables = pd.concat(nuc_tables).sort_values("label")
        cell_all_tables = pd.concat(cell_tables).sort_values("label")

        def rename_vars(x, marker_map, compartment):
            if 'intensity_median' in x:
                index = int(x.split("-")[-1])
                channel_name = marker_map[index]
                return f"{channel_name}: {compartment}: Median"
            elif 'intensity_mean' in x:
                index = int(x.split("-")[-1])
                channel_name = marker_map[index]
                return f"{channel_name}: {compartment}: Mean"
            else:
                return x

        def annotate_marker_names(table, compartment):
            # 1) Rename variables to marker names
            markers_index_map = self.marker_map
            index_markers_map = {k:v for v,k in markers_index_map.items()}

            table.columns = table.columns.map(
                lambda x: rename_vars(x, index_markers_map, compartment))
            return table
        
        def annotate_tmas(table):
            # 2) Map indices to label a cell belonging to the respective TMA
            mapping = self.label_map
            # Create an IntervalIndex from the mapping keys for efficient indexing
            interval_index = pd.IntervalIndex.from_tuples(list(mapping.keys()), closed="both")
            # Prepare labels for the intervals (ensure the order matches the interval_index)
            labels = list(mapping.values())
            # Use pd.cut to assign each 'label' value to an interval, and map intervals to TMA labels
            table['TMA'] = pd.cut(
                table['label'], 
                bins=interval_index, 
                labels=labels, 
                include_lowest=True)
            interval_to_label = {
                interval: label for interval, label in 
                zip(interval_index, mapping.values())}
            # This 'df['TMA']' column will now have the corresponding TMA labels
            table['TMA'] = table['TMA'].map(interval_to_label)
            return table
        
        # Format tables
        # Have columns as Marker: Compartment: Median/Mean 
        nuc_all_tables = annotate_marker_names(nuc_all_tables, "Nucleus")
        cell_all_tables = annotate_marker_names(cell_all_tables, "Cell")
        # Annotate cells with the TMA core they belong to depending on their label index
        nuc_all_tables = annotate_tmas(nuc_all_tables)
        cell_all_tables = annotate_tmas(cell_all_tables)
        # Annotate uuids; at the moment labels match; so can use uuid for one table set
        #uuid_map = dict(zip(nuc_all_tables["label"], nuc_all_tables["uuid"]))

        # Include all metadata in obs
        if mode == "median":
            nuc_marker_columns = nuc_all_tables.columns[nuc_all_tables.columns.str.contains(": Median|label")]
            nuc_meta_columns = nuc_all_tables.columns[~nuc_all_tables.columns.str.contains(": Median")]
            cell_marker_columns = cell_all_tables.columns[cell_all_tables.columns.str.contains(": Median|label")]
            cell_meta_columns = cell_all_tables.columns[~cell_all_tables.columns.str.contains(": Median")]
        else:
            nuc_marker_columns = nuc_all_tables.columns[nuc_all_tables.columns.str.contains(": Mean|label")]
            nuc_meta_columns = nuc_all_tables.columns[~nuc_all_tables.columns.str.contains(": Mean")]
            cell_marker_columns = cell_all_tables.columns[cell_all_tables.columns.str.contains(": Mean|label")]
            cell_meta_columns = cell_all_tables.columns[~cell_all_tables.columns.str.contains(": Mean")]

        # Merge metadata columns; rename metadata to have compartment formatting
        all_meta = pd.merge(
            nuc_all_tables[nuc_meta_columns], 
            cell_all_tables[cell_meta_columns], 
            on="label", 
            suffixes=("-Nucleus", "-Cell"))

        all_expr = pd.merge(
            nuc_all_tables[nuc_marker_columns], 
            cell_all_tables[cell_marker_columns], 
            on="label", 
            suffixes=("-Nucleus", "-Cell"))

        #image_model_init_params = self.sdata.tables[f"{self.image_name}_adata"].uns["image_model_init_params"]
        #var_init = self.sdata.tables[f"{self.image_name}_adata"].var
        # all_expr.to_csv(self.tmp_expr_path)
        # all_meta.to_csv(self.tmp_meta_path)

        # # Cache temp csv to image layer path
        # # NOTE: in the future, image_layer should be a layer group
        # # Where in the layer group is the main image, polygons, masks, etc
        # self.image_layer.metadata["tmp_expr_path"] = self.tmp_expr_path
        # self.image_layer.metadata["tmp_meta_path"] = self.tmp_meta_path
        # TODO: crashes when trying to create AnnData object
        # Remove uuid
        cell_marker_columns = list(cell_marker_columns)
        # cell_marker_columns.remove("uuid")
        cell_marker_columns.remove("label")
        cell_marker_columns.remove("tma_label") # tma label is included

        nuc_marker_columns = list(nuc_marker_columns)
        # nuclear_marker_columns.remove("uuid")
        nuc_marker_columns.remove("label")
        nuc_marker_columns.remove("tma_label") # tma label is included

        print(cell_marker_columns)
        print("trying to create adata")
        adata = ad.AnnData(
            cell_all_tables[cell_marker_columns].values,
            obs=all_meta,
            var=pd.DataFrame(
                list(self.marker_map.keys()),
                columns=["Marker"])
        )
        adata.var_names = adata.var.iloc[:, 0] # TODO maybe parse above better
            #layers={"nuclear":nuc_all_tables[nuc_marker_columns].values})

        # stick to one tma_label
        adata.obs = adata.obs.drop(
            columns=["tma_label-Nucleus", "TMA-Nucleus", "TMA-Cell"])
        adata.obs = adata.obs.rename({"tma_label-Cell":"tma_label"}, axis=1)

        adata.layers["raw_X"] = adata.X
        adata.layers["raw_nuclear_X"] = nuc_all_tables[nuc_marker_columns].values
        
        adata.obsm["spatial"] = adata.obs[
            ["centroid_x_global-Cell", "centroid_y_global-Cell"]
            ]

        self.adata = adata
    
    # def export_to_adata(self, save_path=None):
    #     """ DEPRACATED """
    #     from tmaprocessor.readers._csv_to_adata_reader import load_csv_to_adata
    #     adata = load_csv_to_adata(self.tmp_expr_path, self.tmp_meta_path)
    #     if save_path is not None:
    #         adata.write_h5ad(save_path)
    #     else:
    #         return adata
    
        # Get root folder 
    # def export_as_tma_regions(self, sdata):
    #     """ Export TMA regions as spatialdata 'regions':
        
    #         The goal is to go from:
    #             SpatialData:
    #                 ├── Images
    #                 │     ├── 'main_image': MultiscaleSpatialImage[cyx] (c, y, x)
    #                 |
    #                 |── Labels
    #                 │     ├── 'cell_seg': SpatialImage[cyx] (c, y, x)
    #                 |
    #                 |── Shapes
    #                 │     ├── 'tma_cores': ShapesModel[geometry] (geometry)
    #                 │     ├── 'tma_masks': ShapesModel[geometry] (geometry)
    #                 │     ├── 'tma_envelope': ShapesModel[geometry] (geometry)
    #                 |
    #             with coordinate systems:
    #             ▸ 'global', with elements:
    #                     main_image (Images), cell_seg (Labels), 
    #                     tma_cores (Shapes), tma_masks (Shapes), tma_envelope (Shapes)
                    
    #         Use tma_envelope to create regions for each core, to achieve below:
    #         tma_envelope will contain a tma_label column for labelling below.
    #             SpatialData object with:
    #                 ├── Images
    #                 │     ├── 'A-1_image': SpatialImage[cyx] (c, ysub, xsub)
    #                 │     ├── 'A-2_image': SpatialImage[cyx] (c, ysub, xsub)
    #                 │     ├── 'A-3_image': SpatialImage[cyx] (c, ysub, xsub)
    #                 │     ├── 'A-4_image': SpatialImage[cyx] (c, ysub, xsub)
    #                 |
    #                 |── Labels
    #                 │     ├── 'A-1_cell_seg': SpatialImage[cyx] (c, ysub, xsub)
    #                 │     ├── 'A-2_cell_seg': SpatialImage[cyx] (c, ysub, xsub)
    #                 │     ├── 'A-3_cell_seg': SpatialImage[cyx] (c, ysub, xsub)
    #                 │     ├── 'A-4_cell_seg': SpatialImage[cyx] (c, ysub, xsub)
    #                 |
    #                 |── Shapes
    #                 │     ├── 'tma_cores': ShapesModel[geometry] (geometry)
    #                 │     ├── 'tma_masks': ShapesModel[geometry] (geometry)
    #                 │     ├── 'tma_envelope': ShapesModel[geometry] (geometry)

    #             with coordinate systems:
    #             ▸ 'A-1', with elements:
    #                     1_image (Images), A-1_cell_seg (Labels), 
    #                     A-1_tma_cores (Shapes), A-1_tma_masks (Shapes), 
    #                     A-1_tma_envelope (Shapes)
                
    #             ...

    #             ▸ 'global', with elements:
    #                     1_image (Images), A-1_cell_seg (Labels), 
    #                     A-1_tma_cores (Shapes), A-1_tma_masks (Shapes), 
    #                     A-1_tma_envelope (Shapes), 
    #                     ...
    #                     tma_cores (Shapes), tma_masks (Shapes), 
    #                     tma_envelope (Shapes)
    #     """
    #     # should be only reading sdata.. 

    #     shapes_key = "tma_envelope"

    #     def get_tma_shape(sdata, i, shapes_key="tma_envelope"):
    #         geom, tma_label = sdata.shapes[shapes_key].iloc[i]
    #         bounds = [int(x) for x in geom.bounds]
    #         return tma_label, bounds

    #     def add_indiv_tma_image(sdata, tma_label, ymin, xmin, ymax, xmax):
    #         image = sdata.images[self.image_name]["scale0"].image.data
    #         sub_image = image[:, ymin:ymax, xmin:xmax]
    #         sdata.images[tma_label+"_image"] = Image2DModel.parse(
    #             sub_image,
    #             dims=("c", "y", "x"),
    #             transformations={
    #                 tma_label: Translation([ymin, xmin], axes=("y", "x"))
    #                 })

    #     def add_indiv_tma_labels(sdata, tma_label, ymin, xmin, ymax, xmax):
    #         cell_labels = sdata.labels["cell_segmentation"]["scale0"].image.data
    #         nuc_labels = sdata.labels["nuclear_segmentation"]["scale0"].image.data
    #         sub_cell_labels = cell_labels[ymin:ymax, xmin:xmax]
    #         sub_nuc_labels = nuc_labels[ymin:ymax, xmin:xmax]
    #         sdata.labels[tma_label+"_cell_seg"] = Labels2DModel.parse(
    #             sub_cell_labels,
    #             dims=("y", "x"),
    #             transformations={
    #                 tma_label: Translation([ymin, xmin], axes=("y", "x"))
    #                 })
    #         sdata.labels[tma_label+"_nuc_seg"] = Labels2DModel.parse(
    #             sub_nuc_labels,
    #             dims=("y", "x"),
    #             transformations={
    #                 tma_label: Translation([ymin, xmin], axes=("y", "x"))
    #                 })

    #     num_tmas = len(sdata.shapes[shapes_key])

    #     for i in range(1): #range(num_tmas)
    #         tma_label, bounds = get_tma_shape(sdata, i, shapes_key)
    #         ymin, xmin, ymax, xmax = bounds
    #         add_indiv_tma_image(sdata, tma_label, ymin, xmin, ymax, xmax)
    #         add_indiv_tma_labels(sdata, tma_label, ymin, xmin, ymax, xmax)

class TMAMeasurer(MultiScaleImageOperations):
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
        "orientation"
    ]

    """
    Measure statistics of cells in TMA cores.

    By default the reference intensity image is done on the highest resolution
    (scale0).

    Can wrap clesperanto / napari-regionprops / existing implementations .. ?
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def measure_labels(
        self,
        labels: DataArray, # Instance Labels
        parent_anndata: ad.AnnData,
        exported_table_name: str,
        tiling_shapes: gpd.GeoDataFrame | None = None,
        extended_properties: bool = False,
        intensity_mode: Literal["mean", "median"] = "mean",
        ):

        intensity_image = self.get_image_by_scale("scale0") # Fullscale
        # Validate that labels has the same x and y dims as intensity_image
        if labels.shape != intensity_image.shape[1:]: #Exclude channel c dim, yx only
            raise ValueError(
                "Labels and intensity image must have the same shape.")
        
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
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
            labels = labels.transpose("x", "y")
            intensity_image = intensity_image.transpose("x", "y", "c")
            if intensity_mode == "mean":
                label_props_table = skimage.measure.regionprops_table(
                    labels.data.compute(), # DataArray + Dask -> np.array
                    intensity_image=intensity_image.data.compute(),
                    properties=properties,
                )
            elif intensity_mode == "median":
                properties.remove("intensity_mean")
                label_props_table = skimage.measure.regionprops_table(
                    labels.data.compute(), # DataArray + Dask -> np.array
                    intensity_image=intensity_image.data.compute(),
                    properties=properties,
                    extra_properties=(intensity_median,)
                )
            else:
                raise ValueError("Unsupported intensity aggregation method.")
            
            label_props_table = pd.DataFrame(label_props_table)\

            # Extract the intensities as expression data
            intensities = label_props_table.filter(like="intensity", axis=1)
            obs_like = label_props_table.drop(columns=intensities.columns)
            obs_like = obs_like.rename(columns={"label": "index"})
            
            return intensities, obs_like

        if tiling_shapes is None:
            intensities, obs_like = _measure_intensities_in_labels(
                labels,
                intensity_image,
                properties,
                intensity_mode,
            )

        # Tiled regionprops
        else:
            geoms = tiling_shapes["geometry"]
            bboxes = [x.bounds for x in geoms]
            bboxes_rast = [[int(z) for z in x] for x in bboxes]

            # Prepare tiles
            intensity_tiles = []
            label_tiles = []

            for bbox in bboxes_rast:
                xmin, ymin, xmax, ymax = bbox
                intensity_tiles.append(
                    intensity_image.isel(
                        x=slice(xmin, xmax+1),
                        y=slice(ymin, ymax+1)
                    )
                )
                label_tiles.append(
                    labels.isel(
                        x=slice(xmin, xmax+1),
                        y=slice(ymin, ymax+1)
                    )
                )
            
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
                intensity_tables.append(sub_intensities)
                obs_tables.append(sub_obs_like)
            
            # Merge tables
            intensities = pd.concat(intensity_tables)
            obs_like = pd.concat(obs_tables)

        # Consolidate results
        # Extract channel information from the intensity image, assumed to be 
        # our dataarray
        channel_names = intensity_image.coords["c"].values
        channel_map = {i: name for i, name in enumerate(channel_names)}
        intensities.columns = intensities.columns.map(
            lambda x: channel_map[int(x.split("-")[-1])]
        ) # convert intensity_*-0 to DAPI, etc.

        # If AnnData is existing..?
        if parent_anndata.shape[1] != 0:
            raise ValueError("Parent AnnData must be featureless.")
        
        # If AnnData is featureless
        else:
            # Check if every cell / obj is measured
            assert intensities.shape[0] == parent_anndata.shape[0] 
            assert obs_like.shape[0] == parent_anndata.shape[0]
            # Inherit
            previous_obs = parent_anndata.obs
            previous_uns = parent_anndata.uns

            merged_obs = previous_obs.merge(
                obs_like,
                how="inner",
                on="index"
            )
            merged_obs = merged_obs.rename(
                columns={
                    "centroid-0": "centroid_x",
                    "centroid-1": "centroid_y"
                    }
                )
            
            new_var = pd.DataFrame(
                index=pd.Series(intensities.columns, name="Protein")
                )
            new_var["intensity_mode"] = intensity_mode
            
            adata = ad.AnnData(
                intensities.values,
                obs=merged_obs,
                obsm={
                    "spatial": merged_obs[["centroid_x", "centroid_y"]].values},
                var=new_var,
                uns=previous_uns
            )

        adata_parsed = TableModel.parse(adata)
        
        self.overwrite_element(self.sdata, adata_parsed, exported_table_name)

        #TODO: update sdata of labels layer -> see widget class for details


                



