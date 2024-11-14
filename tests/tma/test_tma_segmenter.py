# also import 'private' classes


def test_segmenter_model_selection():
    """Test order of model selection. Should be base_model, denoise_model,
    then custom file path.
    """
