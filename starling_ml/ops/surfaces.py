"""Чистые операции построения поверхностей из 3D volumes."""


def marching_cubes(volume, level=0.5, spacing=(1.0, 1.0, 1.0)):
    """Строит triangle mesh; scikit-image остаётся optional dependency."""
    try:
        from skimage.measure import marching_cubes as _marching_cubes
    except ImportError as exc:
        raise ImportError("BuildSurface requires scikit-image") from exc

    return _marching_cubes(volume, level=level, spacing=spacing)[:2]
