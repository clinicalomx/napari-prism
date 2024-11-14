from typing import Any

# also import 'private' classes
import numpy as np
import pytest
from pytest_lazy_fixtures import lf
from xarray import DataArray


# test main SdataImageOperations behaviours
def test_multichannel_image_projection(
    singlescale_3d_image_dataarray: Any, sdata_image_operations_stateless: Any
) -> None:
    CHANNELS = ["c1", "c2"]
    out_max = (
        sdata_image_operations_stateless.get_multichannel_image_projection(
            singlescale_3d_image_dataarray, CHANNELS, "max"
        )
    )
    out_max_expected = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    out_mean = (
        sdata_image_operations_stateless.get_multichannel_image_projection(
            singlescale_3d_image_dataarray, CHANNELS, "mean"
        )
    )
    out_mean_expected = np.array(
        [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
    )

    out_sum = (
        sdata_image_operations_stateless.get_multichannel_image_projection(
            singlescale_3d_image_dataarray, CHANNELS, "sum"
        )
    )

    out_median = (
        sdata_image_operations_stateless.get_multichannel_image_projection(
            singlescale_3d_image_dataarray, CHANNELS, "median"
        )
    )

    assert np.all(out_max.data == out_max_expected)
    assert np.all(out_sum.data == out_max_expected)
    assert np.all(out_mean.data == out_mean_expected)
    assert np.all(out_median.data == out_mean_expected)

    assert out_max.coords["c"].data == "max[c1_c2]"
    assert out_sum.coords["c"].data == "sum[c1_c2]"
    assert out_mean.coords["c"].data == "mean[c1_c2]"
    assert out_median.coords["c"].data == "median[c1_c2]"

    assert isinstance(out_max, DataArray)
    assert isinstance(out_sum, DataArray)
    assert isinstance(out_mean, DataArray)
    assert isinstance(out_median, DataArray)


@pytest.mark.parametrize(
    "overwrite_target, input_element, expected",
    [
        ("blobs_labels", lf("labels_model_ones"), "labels/blobs_labels"),
        ("blobs_image", lf("image_model_ones"), "images/blobs_image"),
        # TODO: blobs_shapes -> Circles/Rects/Points
        # TODO: blobs_points -> Points
        # TODO: tables/table -> TableModel
    ],
)
def test_overwrite_element_disk(
    sdata_image_operations_stateless: Any,
    sdata_disk: Any,
    overwrite_target: Any,
    input_element: Any,
    expected: Any,
) -> None:
    # Test backed elements
    assert overwrite_target in sdata_disk
    sdata_image_operations_stateless.overwrite_element(
        sdata_disk, input_element, overwrite_target
    )
    assert expected in sdata_disk.elements_paths_on_disk()
    assert sdata_disk[overwrite_target].shape == input_element.shape


def test_save_element_disk(
    sdata_image_operations_stateless: Any,
    sdata_disk: Any,
    labels_model_ones: Any,
) -> None:
    # Test backed elements
    sdata_image_operations_stateless.overwrite_element(
        sdata_disk, labels_model_ones, "blobs_labels_new"
    )
    assert sdata_disk["blobs_labels_new"].shape == (3, 3)
    assert "blobs_labels" in sdata_disk
    assert "labels/blobs_labels_new" in sdata_disk.elements_paths_on_disk()


@pytest.mark.parametrize(
    "image_label",
    [
        lf("singlescale_2d_image_dataarray"),
        lf("singlescale_2d_image_dask"),
        lf("singlescale_2d_image_numpy"),
    ],
)
def test_add_label_disk(
    image_label: Any,
    sdata_image_operations_disk: Any,
) -> None:
    # reset dims too
    sdata_image_operations_disk.add_label(
        image_label, "img", True, dims=("x", "y")
    )
    assert "img" in sdata_image_operations_disk.sdata


@pytest.mark.parametrize(
    "image",
    [
        lf("singlescale_3d_image_dataarray"),
        lf("singlescale_3d_image_dask"),
        lf("singlescale_3d_image_numpy"),
    ],
)
def test_add_image_disk(image: Any, sdata_image_operations_disk: Any) -> None:
    sdata_image_operations_disk.add_image(
        image, "img", True, dims=("c", "x", "y")
    )
    assert "img" in sdata_image_operations_disk.sdata


def test_add_shapes_disk(sdata_disk: Any) -> None:
    pass


def test_add_table_disk(sdata_disk: Any) -> None:
    pass


# Test labels and linked labels via regions
def test_add_linked_label_table(sdata_disk: Any) -> None:
    pass
