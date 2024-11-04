# tmaprocessor

<!--
#FUTURE: package logo
-->

[![License MIT](https://img.shields.io/pypi/l/tmaprocessor.svg?color=green)](https://github.com/clinicalomx/tmaprocessor/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/tmaprocessor.svg?color=green)](https://pypi.org/project/tmaprocessor)
[![Python Version](https://img.shields.io/pypi/pyversions/tmaprocessor.svg?color=green)](https://python.org)
[![tests](https://github.com/clinicalomx/tmaprocessor/workflows/tests/badge.svg)](https://github.com/clinicalomx/tmaprocessor/actions)
[![codecov](https://codecov.io/gh/clinicalomx/tmaprocessor/branch/main/graph/badge.svg)](https://codecov.io/gh/clinicalomx/tmaprocessor)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/tmaprocessor)](https://napari-hub.org/plugins/tmaprocessor)

### NOTE: This package is still in heavy development.

An end-to-end, interactive and integrated solution for processing and analysing
multiplexed tissue microarray images.

At its core a Python package and [napari] plugin, harnesses the Python 
bioimaging and bioinformatics ecosystem to perform highly interactive image 
processing and analysis of multiplexed tissue microarrays in one user window.

This package uses [spatialdata] as the core data framework, allowing for further
downstream analysis that is FAIR (findable, accesible, interoperable and
reusable). To allow for processing and analysis capabilities within an
interactive GUI window, the package is built on top of [napari] and 
[napari-spatialdata]. 

Currently, end-to-end capabilities are available for images generated from the 
Akoya Phenocycler™-Fusion platform. However, the modular structure of the 
package allows for entry points at any stage of processing and analysis given a 
pre-built SpatialData object using readers from either 
[spatialdata-io] or [sopa].


<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

<!--
## Installation

You can install `tmaprocessor` via [pip]:

    pip install tmaprocessor



To install latest development version :

    pip install git+https://github.com/clinicalomx/tmaprocessor.git
-->

## Getting Started
**FUTURE-documentation
**FUTURE-tutorials

## Contributing

With future improvements and contributions in mind, the package has been 
designed as a loose MVVM datamodel, where the computations and functions are 
performed by base model classes. The View class is the napari viewer instance. 
Each model class has a corresponding widget (ViewModel) class which provides the 
bridge between the computation logic and the napari viewer. The relations can be 
found in the following schema: ***FUTURE

Hence, contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"tmaprocessor" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

## Citation

**tba


[napari]: https://github.com/napari/napari
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt

[file an issue]: https://github.com/clinicalomx/tmaprocessor/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/

[spatialdata]: https://github.com/scverse/spatialdata/tree/main
[napari-spatialdata]: https://github.com/scverse/napari-spatialdata/tree/main
[spatialdata-io]: https://github.com/scverse/spatialdata-io
[sopa]: https://github.com/gustaveroussy/sopa