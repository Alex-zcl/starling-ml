"""Runtime-обёртки операций построения 3D surfaces."""

from ...core.module import Module
from ...ops.surfaces import marching_cubes


class BuildSurface(Module):
    """Читает volume из Context и сохраняет vertices/faces отдельно."""

    def setup(
        self,
        volume,
        vertices="surface.vertices",
        faces="surface.faces",
        level=0.5,
        spacing=(1.0, 1.0, 1.0),
    ):
        self.volume = volume
        self.vertices = vertices
        self.faces = faces
        self.level = level
        self.spacing = spacing

    def __call__(self):
        volume = self.context[self.volume]
        if hasattr(volume, "detach"):
            volume = volume.detach().cpu().numpy()
        vertices, faces = marching_cubes(volume, self.level, self.spacing)
        self.context[self.vertices] = vertices
        self.context[self.faces] = faces
