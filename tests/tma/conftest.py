import shutil

import dask.array as da
import napari
import numpy as np
import pytest
from dask.array import Array
from geopandas import GeoDataFrame
from numpy import ndarray
from spatialdata import SpatialData
from spatialdata.datasets import blobs
from spatialdata.models import Image2DModel, Labels2DModel
from xarray import DataArray

from napari_prism.models.tma_ops._tma_image import SdataImageOperations
from napari_prism.widgets.tma_ops._base_widgets import SdataImageNapariWidget

DEFAULT_COORDS = ("y", "x")


# data
@pytest.fixture(scope="module")
def singlescale_2d_image_numpy() -> ndarray:
    return np.ones((3, 3))


@pytest.fixture(scope="module")
def singlescale_2d_image_dataarray(singlescale_2d_image_numpy) -> DataArray:
    return DataArray(
        singlescale_2d_image_numpy,
        dims=DEFAULT_COORDS,
        coords={
            DEFAULT_COORDS[0]: np.arange(3),
            DEFAULT_COORDS[1]: np.arange(3),
        },
    )


@pytest.fixture(scope="module")
def singlescale_2d_image_dask(singlescale_2d_image_numpy) -> Array:
    return da.array(singlescale_2d_image_numpy)


@pytest.fixture(scope="module")
def labels_model_ones(singlescale_2d_image_dataarray) -> Labels2DModel:
    return Labels2DModel.parse(singlescale_2d_image_dataarray)


@pytest.fixture(scope="module")
def singlescale_3d_image_numpy() -> ndarray:
    c1 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])

    c2 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    c3 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])

    c = np.stack([c1, c2, c3], axis=-1)
    return c


@pytest.fixture(scope="module")
def singlescale_3d_image_dataarray(singlescale_3d_image_numpy) -> DataArray:
    xr = DataArray(
        singlescale_3d_image_numpy,
        dims=["y", "x", "c"],
        coords={"y": np.arange(3), "x": np.arange(3), "c": ["c1", "c2", "c3"]},
    )

    return xr


@pytest.fixture(scope="module")
def singlescale_3d_image_dask(singlescale_3d_image_numpy) -> Array:
    return da.array(singlescale_3d_image_numpy)


@pytest.fixture(scope="module")
def image_model_ones(singlescale_3d_image_dataarray) -> Image2DModel:
    return Image2DModel.parse(singlescale_3d_image_dataarray)


@pytest.fixture(scope="module")
def shapes_gdf() -> GeoDataFrame:
    pass


@pytest.fixture(scope="module")
def table_model(singlescale_3d_image_dataarray) -> Image2DModel:
    return Image2DModel.parse(singlescale_3d_image_dataarray)


@pytest.fixture()
def sdata_memory() -> SpatialData:
    return blobs()


@pytest.fixture()
def viewer_and_image_layer_with_spatialdata(
    make_napari_viewer, sdata_memory
) -> napari.viewer.Viewer:
    viewer = make_napari_viewer()
    sdata_blobs = sdata_memory
    groups = list(sdata_blobs["blobs_multiscale_image"])
    images = [sdata_blobs["blobs_multiscale_image"][s].image for s in groups]
    viewer.add_image(
        images,
        name="blobs_multiscale_image",
        rgb=False,
        multiscale=True,
        metadata={
            "sdata": sdata_blobs,
            "name": "blobs_multiscale_image",
            "adata": None,
        },
    )
    return viewer


# Below mimicks layers added via napari_spatialdata
# Taken from napari_spatialdata/tests/test_widgets.py
# Repurposes the test function as a fixture
@pytest.fixture(scope="module")
def viewer_and_labels_layer_with_spatialdata(
    make_napari_viewer, sdata_memory
) -> napari.viewer.Viewer:
    viewer = make_napari_viewer()
    sdata_blobs = sdata_memory
    viewer.add_labels(
        sdata_blobs["blobs_labels"],
        name="blobs_labels",
        metadata={
            "sdata": sdata_blobs,
            "name": "blobs_labels",
            "adata": sdata_blobs["table"],
            "region_key": sdata_blobs["table"].uns["spatialdata_attrs"][
                "region_key"
            ],
            "instance_key": sdata_blobs["table"].uns["spatialdata_attrs"][
                "instance_key"
            ],
            "table_names": ["table"],
        },
    )
    return viewer


@pytest.fixture(scope="session")
def sdata_disk():
    sdata = blobs()
    temp_sdata_file_name = "test.zarr"  # = tempfile.NamedTemporaryFile(delete=False, suffix=".zarr")
    sdata.write(temp_sdata_file_name)
    yield sdata
    # os.remove(temp_sdata_file_name)
    shutil.rmtree(temp_sdata_file_name)  # folder -> needs recursive remove


# classes
@pytest.fixture(scope="session")
def sdata_image_operations_disk(sdata_disk) -> SdataImageOperations:
    return SdataImageOperations(sdata_disk, "blobs_image")


@pytest.fixture(scope="module")
def sdata_image_operations_memory(sdata_memory) -> SdataImageOperations:
    return SdataImageOperations(sdata_memory, "blobs_image")


# for testing 'static-like' methods
@pytest.fixture(scope="module")
def sdata_image_operations_stateless() -> SdataImageOperations:
    sdata = SpatialData()
    return SdataImageOperations(sdata, "none")


@pytest.fixture(scope="module")
def sdata_image_widget() -> SdataImageNapariWidget:
    pass


@pytest.fixture(scope="module")
def sdata_image_element() -> Image2DModel:
    pass
