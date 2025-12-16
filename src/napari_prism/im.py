# from .models.tma_ops._tma_image import (
#     TMADearrayer,
#     TMAMasker,
#     TMAMeasurer,
#     TMASegmenter,
#     dearray_tma,
#     mask_tma,
#     measure_tma,
#     segment_tma,
# )

__all__ = [
    "mask_tma",
    "dearray_tma",
    "segment_tma",
    "measure_tma",
    "TMAMasker",
    "TMADearrayer",
    "TMASegmenter",
    "TMAMeasurer",
]

def dearray_tma(*args, **kwargs):
    from .models.tma_ops._tma_image import dearray_tma as _dearray_tma
    return _dearray_tma(*args, **kwargs)

def mask_tma(*args, **kwargs):
    from .models.tma_ops._tma_image import mask_tma as _mask_tma
    return _mask_tma(*args, **kwargs)

def measure_tma(*args, **kwargs):
    from .models.tma_ops._tma_image import measure_tma as _measure_tma
    return _measure_tma(*args, **kwargs)

def segment_tma(*args, **kwargs):
    from .models.tma_ops._tma_image import segment_tma as _segment_tma
    return _segment_tma(*args, **kwargs)

def __getattr__(name):
    if name == "TMADearrayer":
        from .models.tma_ops._tma_image import TMADearrayer
        return TMADearrayer
    if name == "TMAMasker":
        from .models.tma_ops._tma_image import TMAMasker
        return TMAMasker
    if name == "TMAMeasurer":
        from .models.tma_ops._tma_image import TMAMeasurer
        return TMAMeasurer
    if name == "TMASegmenter":
        from .models.tma_ops._tma_image import TMASegmenter
        return TMASegmenter
    raise AttributeError(f"module {__name__} has no attribute {name}")
