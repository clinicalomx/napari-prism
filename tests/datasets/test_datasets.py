from xarray import DataTree

import napari_prism.datasets as datasets


def test_nsclc4301_truncated_exists():
    import napari_prism.datasets as datasets

    # Check if the function exists
    if hasattr(datasets, "nsclc4301_truncated"):
        assert callable(datasets.nsclc4301_truncated)


def test_dataset_download_mock():
    """Test dataset download functionality with mocked requests;

    If error, indicates some issue with zenodo server and github backup
    """
    sdata = datasets.nsclc4301_truncated()
    # check presence
    assert "NSCLC4301" in sdata.images
    assert {"global", "um"} == set(sdata.coordinate_systems)
    # check types
    assert isinstance(sdata["NSCLC4301"], DataTree)  # Should be multiscale
    # should be 5 scales
    assert len(sdata["NSCLC4301"]) == 5
    # and should be the following channels
    EXPECTED_CHANS = [
        "DAPI",
        "E-cadherin",
        "Vimentin",
        "Pan-Cytokeratin",
        "CD20",
        "CD4",
    ]
    assert sdata["NSCLC4301"]["scale0"].c.data.tolist() == EXPECTED_CHANS
