from abc import ABC, abstractmethod
from collections import defaultdict
from io import StringIO
from pathlib import Path

import dask.array as da
import pandas as pd
import tifffile

# from napari_spatialdata import Interactive
import zarr
from dask.delayed import delayed as dd
from loguru import logger
from napari_spatialdata import Interactive
from spatialdata import SpatialData
from spatialdata.models import Image2DModel
from spatialdata.transformations import Identity, Scale


class QptiffLoader(ABC):
    META_ORDER = []

    def __init__(self, qptiff_path, *args, **kwargs):
        self.qptiff_path = qptiff_path
        self.load_page_series(qptiff_path, *args, **kwargs)
        self.generate_metadata()
        self.image_name = self.reformat_image_name(
            Path(qptiff_path).absolute().stem
        )

    def __str__(self):
        string = (
            f"QPTIFF with fields: \n"
            f"  .image: {self.image}\n"
            f"  .markers\n"
            f"  .marker_info\n"
            f"  .scan_profile\n"
            f"  .channel_exposure\n"
            f"  .unusued_items\n"
            f"  .wells\n"
        )
        return string

    def __repr__(self):
        return self.__str__()

    def reformat_image_name(self, input_name):
        return input_name.replace(" ", "_").replace("-", "_")

    @staticmethod
    def series_to_array(series):
        return series.asarray()

    def load_multiscale_image_with_zarr(self, return_cache_list=False):
        with tifffile.imread(self.qptiff_path, aszarr=True) as store:
            # store = zarr.LRUStoreCache(store, max_size=2**30)
            zobj = zarr.open(store, mode="r")
            store.close()
            if return_cache_list:
                data = [
                    zobj[int(dataset["path"])]
                    for dataset in zobj.attrs["multiscales"][0]["datasets"]
                ]
                return data
            else:
                return zobj

    def infer_scales_from_qptiff(self):
        # When we parse our zarr caches or data, as with load multiscale, we can
        # get the respective scaling factors
        zarr_list = self.load_multiscale_image_with_zarr(True)

        def get_shapes(arrays):
            return [array.shape for array in arrays]

        # Function to calculate scaling factors
        def calculate_scaling_factors(shapes):
            scaling_factors = []
            for i in range(len(shapes) - 1):
                x_scale = shapes[i][1] / shapes[i + 1][1]
                y_scale = shapes[i][2] / shapes[i + 1][2]

                # Assume x and y are scaled the same way, take the average
                avg_scale = (x_scale + y_scale) / 2
                scale_factor = round(avg_scale)

                if scale_factor != 1:
                    scaling_factors.append(scale_factor)

            return scaling_factors

        shapes = get_shapes(zarr_list)
        return calculate_scaling_factors(shapes)

    def load_page_series(self, qptiff_path, channel_int=None, with_zarr=False):
        if with_zarr:
            with tifffile.TiffFile(qptiff_path) as tif:
                series = tif.series[0]
                self.page_series = series
            zobj = self.load_multiscale_image_with_zarr()
            z = zobj[int(zobj.attrs["multiscales"][0]["datasets"][0]["path"])]
            self.image = da.from_zarr(
                z
            )  # for z in data] # Can parse these as ImageModel2D's (multi)

        else:
            with tifffile.TiffFile(qptiff_path) as tif:
                series = tif.series[0]
                self.page_series = series
                if channel_int is not None:
                    series = series[channel_int]

                # image = dd(lambda series: series.asarray())(tif)
                delayed_image = dd(self.series_to_array)(series)
                image = da.from_delayed(
                    delayed_image, dtype=series.dtype, shape=series.shape
                )
                self.image = image

    @abstractmethod
    def generate_metadata(self):
        raise NotImplementedError(
            "Calling abstract method from abstract class."
        )

    # Required attributes; have setters
    def set_channel_map(self, channel_map):
        self.channel_map = channel_map

    def set_scaling_factor(self, scaling_factor):
        self.scaling_factor = scaling_factor

    def set_markers(self, markers):
        self.markers = markers

    def set_marker_info(self, marker_info):
        self.marker_info = marker_info
        # self.marker_info.index = self.markers # Need markers

    def to_spatialdata(self, downscale_factors=None):
        """Creates base spatialdata object containing only the multiscale
        from qptiff data."""
        # downscale_factors determine pyramidal downscaling of qptiff img;
        # /2 -> /4 -> /8 -> /16
        if downscale_factors is None:
            downscale_factors = [2, 2, 2, 2]
        transformations = {}
        transformations["global"] = Identity()
        # transformations["scale0"] = Identity()
        transformations["um"] = Scale(
            [self.scaling_factor, self.scaling_factor], axes=("x", "y")
        ).inverse()  # converts pixels to um using qptiff provided metadata for scaling

        if downscale_factors == "infer":
            downscale_factors = self.infer_scales_from_qptiff()

        self.downscale_factors = downscale_factors  # used to match multiscaling of main image for fullscale images produced
        # # For each downscaling factor, provide a coordinate scale with respect to global px coord.
        # prev_s = 1
        # for i, s in enumerate(downscale_factors):
        #     i += 1
        #     ss = s*prev_s
        #     transformations[f"ds_{(2**i)}x"] = Scale([ss, ss], axes=("x", "y")).inverse()
        #     prev_s = ss

        # Parse full image with transformations and names
        # OME-NGFF
        image = Image2DModel.parse(
            self.image,
            dims=("c", "y", "x"),
            chunks=(1, 5120, 5120),
            transformations=transformations,
            c_coords=self.markers,
            scale_factors=downscale_factors,
        )  # Passing scale factors with tuples -> MultiscaleSpatialImage

        # Finally the spatialdata that represents the qptiff file
        sdata = SpatialData(
            images={self.image_name: image},
        )  # ,
        # tables={self.image_name + "_adata": adata_table})

        return sdata


class PCFLoader(QptiffLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_metadata(self):
        data = {}
        for i, page in enumerate(self.page_series):
            pandas_series = pd.read_xml(
                StringIO(page.description), parser="etree"
            ).stack()
            data[i] = pandas_series
        data_df = pd.DataFrame(data).T
        data_df.columns = data_df.columns.droplevel(0)
        scan_profile = pd.read_json(StringIO(data_df["ScanProfile"][0]))
        channel_exposure = pd.DataFrame(
            scan_profile["experimentDescription"]["channels"]
        )
        unused_items = pd.DataFrame(
            scan_profile["experimentDescription"]["unusedItems"]
        )
        wells = pd.DataFrame(scan_profile["experimentDescription"]["wells"])
        scan_profile = scan_profile.drop(columns=["experimentDescription"])
        data_df = data_df.drop(columns=["ScanProfile"])

        def _make_cols_unique(df):
            # by product of xml readers; of repeated fields but in different nestings
            renamer = defaultdict()
            for column_name in df.columns[
                df.columns.duplicated(keep=False)
            ].tolist():
                if column_name not in renamer:
                    renamer[column_name] = [column_name + "_0"]
                else:
                    renamer[column_name].append(
                        column_name + "_" + str(len(renamer[column_name]))
                    )
            return df.rename(
                columns=lambda column_name: (
                    renamer[column_name].pop(0)
                    if column_name in renamer
                    else column_name
                )
            )

        data_df = _make_cols_unique(data_df)

        self.set_markers(data_df["Biomarker"].to_list())
        self.set_marker_info(data_df)
        self.set_scaling_factor(data_df["ScaleFactor"].unique()[0])
        self.set_channel_map(
            {v: k for k, v in data_df.Biomarker.to_dict().items()}
        )
        self.mpp = 1 / self.scaling_factor

        # Misc
        self.scan_profile = scan_profile
        self.channel_exposure = channel_exposure
        self.wells = wells
        self.unused_items = unused_items


def qptiff_reader_function(
    qptiff_path: str | Path, target_path: str | Path | None = None
) -> list[tuple[None]]:
    """Load .qptiff file into a SpatialData format, save to zarr store.

    By default lazily loads .qptiff, stored as multiscale images, with:
        - scale0: Raw image
        - scale1: Downsampled 2x
        - scale2: Downsampled 4x
        - scale3: Downsampled 8x
        - scale4: Downsampled 16x

    Then pipes written zarr store to napari_spatialdata reader.

    """
    logger.info(f"Converting {qptiff_path} to SpatialData")
    sdata = PCFLoader(qptiff_path).to_spatialdata()

    if isinstance(qptiff_path, str):
        qptiff_path = Path(qptiff_path)

    if target_path is not None:
        # Validate correct path
        if isinstance(target_path, str):
            target_path = Path(target_path)
        if target_path.suffix != ".zarr":
            raise ValueError("Target path must be a .zarr file.")

    else:
        target_path = qptiff_path.with_suffix(".zarr")

    logger.info(f"Saving SpatialData to {target_path}")
    sdata.write(target_path)
    logger.info(f"SpatialData zarr store created at {target_path}")

    logger.info(f"Launching napari_spatialdata with {target_path}")
    napari_spatialdata_with_prism_reader(target_path)

    # Then launch TMA plugins
    return [(None,)]


class PrismInteractive(Interactive):
    """Extends napari-spatialdata entry point, adds the plugin widgets on-top."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._viewer.window.add_plugin_dock_widget(
            "napari-prism", "TMA Image Analysis", tabify=True
        )
        self._viewer.window.add_plugin_dock_widget(
            "napari-prism", "AnnData Analysis", tabify=True
        )


def napari_spatialdata_with_prism_reader(path: str | Path):
    """Read .zarr file into napari_spatialdata reader, with TMA plugins"""
    # native_napari_spatialdata_reader(path)
    # iewer = napari.current_viewer()
    # Launch TMA plugin along side..
    # TODO: interative like component
    sdata = SpatialData.read(path)
    _ = PrismInteractive(sdata, headless=True)
    return [(None,)]


def napari_get_reader(path: str):
    """
    Reader with two entry points:

    1) .qptiff -> Gets parsed into a SpatialData object, saved to zarr store.
    2) .zarr -> Reads .zarr spatialdata object via napari-spatialdata

    If starting from 1), gets piped to 2)

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if path.endswith(".qptiff"):
        return qptiff_reader_function

    elif path.endswith(".zarr"):
        return napari_spatialdata_with_prism_reader
    # otherwise we return the *function* that can read ``path``.
    else:
        return None
