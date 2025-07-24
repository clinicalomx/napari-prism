from collections import defaultdict
from io import StringIO
from pathlib import Path

import dask.array as da
import pandas as pd
import tifffile
import zarr
from dask.delayed import delayed as dd
from loguru import logger
from spatialdata import SpatialData
from spatialdata.models import Image2DModel
from spatialdata.transformations import Identity, Scale

from napari_prism.constants import DEFAULT_MULTISCALE_DOWNSCALE_FACTORS


class QptiffReader:
    def __init__(self, qptiff_path, image_index=0, *args, **kwargs):
        self.qptiff_path = qptiff_path
        self.load_page_series(qptiff_path, image_index, *args, **kwargs)
        self.generate_metadata()
        self.image_name = self.reformat_image_name(qptiff_path)

    def __str__(self):
        string = (
            f"QPTIFF with fields: \n"
            f"  .image: {self.image}\n"
            f"  .markers\n"
            f"  .marker_info\n"
        )
        return string

    def __repr__(self):
        return self.__str__()

    def reformat_image_name(self, input_name):
        path = Path(input_name)
        # Remove all path suffixes
        while path.suffix:
            path = path.with_suffix("")

        path = path.stem
        path = path.replace(" ", "_").replace("-", "_")
        return path

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

    def load_page_series(
        self, qptiff_path, image_index=0, channel_int=None, with_zarr=False
    ):
        if with_zarr:
            with tifffile.TiffFile(qptiff_path) as tif:
                series = tif.series[
                    image_index
                ]  # TODO: 0-> Baseline image -> Set as constant
                self.page_series = series
            zobj = self.load_multiscale_image_with_zarr()
            z = zobj[int(zobj.attrs["multiscales"][0]["datasets"][0]["path"])]
            self.image = da.from_zarr(
                z
            )  # for z in data] # Can parse these as ImageModel2D's (multi)

        else:
            with tifffile.TiffFile(qptiff_path) as tif:
                series = tif.series[
                    image_index
                ]  # TODO: 0-> Baseline image -> Set as constant
                self.page_series = series
                if channel_int is not None:
                    series = series[channel_int]

                delayed_image = dd(lambda x: x.asarray())(series)
                image = da.from_delayed(
                    delayed_image, dtype=series.dtype, shape=series.shape
                )
                self.image = image

    def generate_metadata(self):
        data = {}
        for i, page in enumerate(self.page_series):
            ppm_x, ppm_y = page.get_resolution(
                5
            )  # tiffile.py -> 5 is the microns tag
            try:
                pandas_series = pd.read_xml(
                    StringIO(page.description), parser="etree"
                ).stack()
            except ValueError:
                pandas_series = pd.read_xml(
                    StringIO(page.description), parser="etree", xpath="."
                ).stack()
            pandas_series = pandas_series.droplevel(0)
            pandas_series["MPP_x"] = 1 / ppm_x  # TODO; make colname constant
            pandas_series["MPP_y"] = 1 / ppm_y  # TODO; make colname constant
            data[i] = pandas_series
        data_df = pd.DataFrame.from_dict(data, orient="index")

        def _make_cols_unique(df):
            # byproduct of xml readers; of repeated fields but in different nestings
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

        if "Biomarker" not in data_df.columns:
            self.markers = None
            self.channel_map = None
        else:
            self.markers = data_df[
                "Biomarker"
            ].to_list()  # Biomarker field should always be in qpi images
            self.channel_map = {
                v: k for k, v in data_df["Biomarker"].to_dict().items()
            }

        self.marker_info = data_df
        self.set_mpp(data_df, "MPP_x", "MPP_y")
        self.axes = tuple(self.page_series.axes.lower())  # Lower case axes

    def set_mpp(self, data_df, mpp_x, mpp_y):
        if (data_df[mpp_x].nunique() > 1) or (data_df[mpp_y].nunique() > 1):
            raise NotImplementedError(
                "Unhandled physical unit calibration for different scaling"
                "factors across channel axis."
            )
        scaling_x = data_df[mpp_x].mean()
        scaling_y = data_df[mpp_y].mean()
        if scaling_x != scaling_y:
            self.mpp = (scaling_x + scaling_y) / 2
        else:
            self.mpp = scaling_x

    def to_spatialdata(self, downscale_factors=None):
        """Creates base spatialdata object containing only the multiscale
        from qptiff data."""
        # downscale_factors determine pyramidal downscaling of qptiff img;
        # /2 -> /4 -> /8 -> /16
        if downscale_factors is None:
            downscale_factors = DEFAULT_MULTISCALE_DOWNSCALE_FACTORS
        transformations = {}
        transformations["global"] = Identity()
        # transformations["scale0"] = Identity()
        transformations["um"] = Scale(
            [1 / self.mpp, 1 / self.mpp], axes=("x", "y")
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
            dims=self.axes,
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
    sdata = QptiffReader(qptiff_path).to_spatialdata()

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


def prism_interactive(*args, **kwargs):
    """Extends napari-spatialdata entry point, adds the plugin widgets on-top.

    Modified to a factory function to prevent API-only / headless environments
    from importing this (i.e. where Qt bindings may not be available).
    """
    from napari_spatialdata import Interactive

    class PrismInteractive(Interactive):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._viewer.window.add_plugin_dock_widget(
                "napari-spatialdata", "Scatter", tabify=True
            )
            self._viewer.window.add_plugin_dock_widget(
                "napari-prism", "TMA Image Analysis", tabify=True
            )
            self._viewer.window.add_plugin_dock_widget(
                "napari-prism", "AnnData Analysis", tabify=True
            )

    return PrismInteractive(*args, **kwargs)


def qptiff(path: str | Path, save_path: str | Path | None = None):
    """
    Read .qptiff file into a SpatialData object and optionally save as a zarr
    store.

    Args:
        path: Path to the .qptiff file.
        save_path: Path to optionally save an on-disk representation as a
            zarr store.

    Returns:
        The converted SpatialData object.

    """
    sdata = QptiffReader(path).to_spatialdata()
    if save_path is not None:
        sdata.write(save_path)
    return sdata


def ometiff(path: str | Path, save_path: str | Path | None = None):
    """
    Read .ome.tiff file into a SpatialData object and optionally save as a
    zarr store.

    Args:
        path: Path to the .ome.tiff file.
        save_path: Path to optionally save an on-disk representation as a
            zarr store.

    Returns:
        The converted SpatialData object.
    """
    from ome_types import from_xml

    class OmeTiffReader:
        def __init__(self, ome_tiff_path, image_index=0, *args, **kwargs):
            self.ome_tiff_path = ome_tiff_path
            self.load_page_series(ome_tiff_path, image_index, *args, **kwargs)
            self.generate_metadata(image_index)
            self.image_name = self.reformat_image_name(ome_tiff_path)

        def load_page_series(
            self, qptiff_path, image_index=0, channel_int=None, with_zarr=False
        ):
            if with_zarr:
                with tifffile.TiffFile(qptiff_path) as tif:
                    series = tif.series[
                        image_index
                    ]  # TODO: 0-> Baseline image -> Set as constant
                    self.page_series = series
                zobj = self.load_multiscale_image_with_zarr()
                z = zobj[
                    int(zobj.attrs["multiscales"][0]["datasets"][0]["path"])
                ]
                self.image = da.from_zarr(
                    z
                )  # for z in data] # Can parse these as ImageModel2D's (multi)

            else:
                with tifffile.TiffFile(qptiff_path) as tif:
                    series = tif.series[
                        image_index
                    ]  # TODO: 0-> Baseline image -> Set as constant
                    self.page_series = series
                    if channel_int is not None:
                        series = series[channel_int]

                    delayed_image = dd(lambda x: x.asarray())(series)
                    image = da.from_delayed(
                        delayed_image, dtype=series.dtype, shape=series.shape
                    )
                    self.image = image
                    self.ome_meta_obj = from_xml(tif.ome_metadata)

        def set_mpp(self, data_df, mpp_x, mpp_y):
            if (data_df[mpp_x].nunique() > 1) or (
                data_df[mpp_y].nunique() > 1
            ):
                raise NotImplementedError(
                    "Unhandled physical unit calibration for different scaling"
                    "factors across channel axis."
                )
            scaling_x = data_df[mpp_x].mean()
            scaling_y = data_df[mpp_y].mean()
            if scaling_x != scaling_y:
                self.mpp = (scaling_x + scaling_y) / 2
            else:
                self.mpp = scaling_x

        def reformat_image_name(self, input_name):
            path = Path(input_name)
            # Remove all path suffixes
            while path.suffix:
                path = path.with_suffix("")

            path = path.stem
            path = path.replace(" ", "_").replace("-", "_")
            return path

        def generate_metadata(self, image_index=0):
            selected_image_meta = self.ome_meta_obj.images[image_index]
            init_df = (
                pd.read_xml(
                    StringIO(selected_image_meta.to_xml()), parser="etree"
                )
                .stack()
                .droplevel(0)
            )

            init_df = pd.DataFrame(init_df).T

            # Extracrt channel info;
            channel_data = {}
            for i, channel in enumerate(selected_image_meta.pixels.channels):
                channel_data[i] = (
                    pd.read_xml(
                        StringIO(channel.to_xml()), parser="etree", xpath="."
                    )
                    .stack()
                    .droplevel(0)
                )
            channel_df = pd.DataFrame(channel_data).T

            # Physical units
            res_data = {}
            for i, page in enumerate(self.page_series):
                ppm_x, ppm_y = page.get_resolution(5)
                res_data[i] = pd.Series(
                    {
                        "MPP_x": 1 / ppm_x,
                        "MPP_y": 1 / ppm_y,
                    }
                )
            res_df = pd.DataFrame(res_data).T
            channel_df = pd.concat([channel_df, res_df], axis=1)
            merged_df = pd.concat(
                [init_df] * len(channel_df), ignore_index=True
            )
            data_df = pd.concat(
                [merged_df, channel_df.reset_index(drop=True)], axis=1
            )

            # Name is Biomarker in ome tiff;s;
            if "Name" not in data_df.columns:
                self.markers = None
                self.channel_map = None
            else:
                self.markers = data_df[
                    "Name"
                ].to_list()  # Biomarker field should always be in qpi images
                self.channel_map = {
                    v: k for k, v in data_df["Name"].to_dict().items()
                }

            self.marker_info = data_df
            # self.set_mpp(data_df, "MPP_x", "MPP_y")
            self.mpp_x = data_df["MPP_x"].mean()
            self.mpp_y = data_df["MPP_y"].mean()

            self.axes = tuple(self.page_series.axes.lower())  # Lower case axes

        def to_spatialdata(self, downscale_factors=None):
            """Creates base spatialdata object containing only the multiscale
            from qptiff data."""
            # downscale_factors determine pyramidal downscaling of qptiff img;
            # /2 -> /4 -> /8 -> /16
            if downscale_factors is None:
                downscale_factors = DEFAULT_MULTISCALE_DOWNSCALE_FACTORS
            transformations = {}
            transformations["global"] = Identity()
            # transformations["scale0"] = Identity()
            transformations["um"] = Scale(
                [1 / self.mpp_x, 1 / self.mpp_y], axes=("x", "y")
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
                dims=self.axes,
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

    sdata = OmeTiffReader(path).to_spatialdata()
    if save_path is not None:
        sdata.write(save_path)
    return sdata


def napari_spatialdata_with_prism_reader(path: str | Path):
    """Read .zarr file into napari_spatialdata reader, with TMA plugins"""
    # native_napari_spatialdata_reader(path)
    # iewer = napari.current_viewer()
    # Launch TMA plugin along side..
    # TODO: interative like component
    sdata = SpatialData.read(path)
    _ = prism_interactive(sdata, headless=True)
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
