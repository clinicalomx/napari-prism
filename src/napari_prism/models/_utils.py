def overrides(interface_class):
    def overrider(method):
        assert method.__name__ in dir(interface_class)
        return method

    return overrider


def overwrite_element(sdata, element_name):
    """Workaround 1
    https://github.com/scverse/spatialdata/blob/main/tests/io/test_readwrite.py
    """
    # backup copy
    new_name = element_name + "_new_place"
    sdata[new_name] = sdata[element_name]
    sdata.write_element(new_name)

    # delete original,
    sdata.delete_element_from_disk(element_name)
    sdata.write_element(element_name)
    del sdata[new_name]
    sdata.delete_element_from_disk(new_name)
