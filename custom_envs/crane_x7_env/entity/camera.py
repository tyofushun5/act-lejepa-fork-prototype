import numpy as np

from .entity import Entity


class ObsCamera(Entity):
    '''Observation camera (same default viewpoint as the reference).'''

    def __init__(self, scene=None, res=(128, 128), pos=(1.0, 1.0, 0.10),
                 lookat=(0.200, 0.0, 0.10), fov=30.0):
        super().__init__(scene=scene)
        self.res = tuple(res)
        self.pos = tuple(pos)
        self.lookat = tuple(lookat)
        self.fov = float(fov)
        self.cam = None

    def create(self):
        self.cam = self.scene.add_camera(
            res=self.res, pos=self.pos, lookat=self.lookat, fov=self.fov, GUI=False,
        )
        return self.cam

    def get_image(self):
        rgb, *_ = self.cam.render()
        return np.asarray(rgb, dtype=np.uint8)
