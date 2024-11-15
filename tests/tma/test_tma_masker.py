import pytest
from spatialdata import SpatialData

from napari_prism.models.tma_ops._tma_image import TMAMasker
from napari_prism.widgets.tma_ops._tma_image_widgets import (
    TMAMaskerNapariWidget,
)

# inputs -> Sdta image_name, reference_coordinate_system
WIDGET_LIST = [
    "_image_shape_selection_widget",
    "_bbox_shape_layer_selection",
    "_sigma_slider",
    "_expansion_slider",
    "_sobel_button",
    "_adaptive_histogram_button",
    "_gamma_correction_button",
    "_estimated_core_diameter_um_entry",
    "_thresholding_core_fraction_entry",
    "_generate_masks_button",
]


@pytest.fixture(scope="function")
def masker_model(sdata_disk) -> TMAMasker:
    TEST_IMAGE_NAME = "blobs_multiscale_image"
    masker_model = TMAMasker(sdata_disk, TEST_IMAGE_NAME)
    return masker_model


def test_init(masker_model, make_napari_viewer):
    # Model init
    assert masker_model is not None
    assert isinstance(masker_model.sdata, SpatialData)

    # Widget init
    viewer = make_napari_viewer()
    widget = TMAMaskerNapariWidget(viewer, masker_model)
    assert widget is not None

    # Check widgets are there and displayed
    for wname in WIDGET_LIST:
        attr = getattr(widget, wname, False)
        assert attr

        # Should have a null shape selection
        if wname == "_image_shape_selection_widget":
            assert attr.choices == (
                (
                    3,
                    128,
                    128,
                ),  # ms widget should show the smallest scale first
                (3, 256, 256),
                (3, 512, 512),
            )


def test_init_empty(make_napari_viewer):
    # The widget can take a null masker when i.e. launching the plugin
    viewer = make_napari_viewer()
    widget = TMAMaskerNapariWidget(viewer, None)
    assert widget is not None
    assert widget.model is None

    # Check widgets are there and displayed
    for wname in WIDGET_LIST:
        attr = getattr(widget, wname, False)
        assert attr

        # Should have a null shape selection
        if wname == "_image_shape_selection_widget":
            assert attr.choices == (None,)


def test_update_model(viewer_and_image_layer_with_spatialdata):
    viewer = viewer_and_image_layer_with_spatialdata

    widget = TMAMaskerNapariWidget(viewer, None)
    assert widget is not None
    assert widget.model is None

    widget.update_model()
    assert widget.model is not None
    # Points to the same sdata mem reference
    assert widget.model.sdata == viewer.layers[-1].metadata["sdata"]
    assert widget.model.image_name == viewer.layers[-1].metadata["name"]

    # Then check that the respective buttons have been enabled
    assert widget._generate_masks_button.enabled is True
    # And that the updated sdata attrs are shown to the user
    attr = getattr(widget, "_image_shape_selection_widget", False)
    assert attr
    assert attr.choices == (
        (3, 128, 128),  # ms widget should show the smallest scale first
        (3, 256, 256),
        (3, 512, 512),
    )


# def test_bbox_layer_selection(masker_model, make_napari_viewer):
#     # Widget init
#     viewer = make_napari_viewer()
#     widget = TMAMaskerNapariWidget(viewer, masker_model)
#     assert widget._bbox_shape_layer_selection.choices == (None,)

#     # THen when an appropraite layer is added, should display
#     viewer.add_
