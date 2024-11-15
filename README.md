# PRISM: A **P**ython package for the inte**R**active and **I**ntegrated analy**S**is of **M**ultiplexed tissue microarrays

<!--
#FUTURE: package logo
-->

[![License MIT](https://img.shields.io/pypi/l/napari-prism.svg?color=green)](https://github.com/clinicalomx/napari-prism/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-prism.svg?color=green)](https://pypi.org/project/napari-prism)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-prism.svg?color=green)](https://python.org)
[![tests](https://github.com/clinicalomx/napari-prism/workflows/tests/badge.svg)](https://github.com/clinicalomx/napari-prism/actions)
[![codecov](https://codecov.io/gh/clinicalomx/napari-prism/branch/main/graph/badge.svg)](https://codecov.io/gh/clinicalomx/napari-prism)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-prism)](https://napari-hub.org/plugins/napari-prism)

**NOTE: PRISM is still in heavy development.**
PRISM or napari-prism is a package and [napari] plugin designed for interactive processing, analysing and visualising multiplxed tissue microarrays.

Currently, end-to-end capabilities (i.e. starting from importing the raw image file, to basic spatial analysis of annotated cells) are available for images generated from the
Akoya Phenocycler™-Fusion platform. However, the modular structure of the
package allows for usage at any stage of processing and/or analysis, given a pre-built SpatialData object using readers from either
[spatialdata-io] or [sopa].

PRISM uses [spatialdata] as the core data framework, allowing for:

1. The rich integration of tools from the ([scverse]) Python bioinformatics ecosystem with highly interactive graphical user interfaces from [napari] and [napari-spatialdata].
2. The storage of images, shapes, annotations and their linked `AnnData` objects in a standardized, FAIR-compliant data structure, addressing the non-standard and fragmented organization of files before, during, and after a multiplexed image analysis pipeline.

The package was designed to be used completely within the [napari] application and therefore require little to no knowledge of Python programming. Therefore, documentation for usage via the API is currently in progress.

## Installation: CPU only

Install this package via [pip]:

```bash
pip install napari-prism
```

Install the latest development version:

```bash
pip install git+https://github.com/clinicalomx/napari-prism.git@main
```

## Installation: GPU-accelerated

### General computations with RAPIDS and rapids-singlecell

General larger scale and/or computationally demanding functions can be accelerated with the [NVIDIA RAPIDS suite](https://rapids.ai/). We utilise some packages from this suite, as well as the GPU-accelerated implementation of scanpy with [rapids-singlecell].

1. [Check and configure the system requirements from RAPIDS](https://docs.rapids.ai/install/#system-req).
    - Currently, only Linux distributions (or Windows systems with WSL2) are supported.
    - Install the [CUDA12.2](https://developer.nvidia.com/cuda-12-2-2-download-archive) or [CUDA12.5](https://developer.nvidia.com/cuda-12-5-1-download-archive) toolkit.
2. Install the package together with [RAPIDS] and [rapids-singlecell] via [pip]:

```bash
pip install napari-prism[gpu] --extra-index-url=https://pypi.nvidia.com
```

### Cell segmentation with Cellpose

To run [cellpose] on the GPU, install the [CUDA version of PyTorch](https://pytorch.org/get-started/locally/). You may need to [remove any installed CPU versions of PyTorch](https://github.com/MouseLand/cellpose?tab=readme-ov-file#gpu-version-cuda-on-windows-or-linux).

## Getting Started

To start using `napari-prism`, please see the [tutorials](https://napari-prism.readthedocs.io/en/latest/notebooks/getting_started.html#):

-   [Getting started](https://napari-prism.readthedocs.io/en/latest/notebooks/getting_started.html)
-   To learn how to interactively analyse raw .qptiff TMAs, see [TMA Image Analysis](https://napari-prism.readthedocs.io/en/latest/notebooks/tma_usage.html)
-   To learn how to interactively analyse AnnData-contained SpatialData objects, see [Anndata Analysis](https://napari-prism.readthedocs.io/en/latest/notebooks/adata_usage.html)

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"napari-prism" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

## Citation

\*\*tba

[napari]: https://github.com/napari/napari
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[file an issue]: https://github.com/clinicalomx/napari-prism/issues
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
[PyTorch]: https://pytorch.org/
[cellpose]: https://github.com/MouseLand/cellpose
[RAPIDS]: https://rapids.ai/
[rapids-singlecell]: https://github.com/scverse/rapids_singlecell
[spatialdata]: https://github.com/scverse/spatialdata/tree/main
[napari-spatialdata]: https://github.com/scverse/napari-spatialdata/tree/main
[spatialdata-io]: https://github.com/scverse/spatialdata-io
[sopa]: https://github.com/gustaveroussy/sopa
[scverse]: https://scverse.org/
