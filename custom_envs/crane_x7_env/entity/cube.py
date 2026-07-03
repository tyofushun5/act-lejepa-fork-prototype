import numpy as np

from .entity import Entity


class Cube(Entity):
    '''Graspable cube.

    Fix vs the reference: `gs.morphs.Box(pos=...)` takes the box CENTER, so we
    pass the center directly instead of the reference's `center - half`
    (which buried the cube half a size into the floor and offset it in xy).
    '''

    def __init__(self, scene=None, surface=None, center=None,
                 size=0.025, color=(0.45, 0.45, 0.45), friction=2.0):
        super().__init__(scene=scene, surface=surface)
        self.size = float(size)
        self.half = self.size / 2
        if center is None:
            center = (0.30, 0.0, self.half + 1e-3)
        self.center = np.array(center, dtype=np.float64)
        self.color = color
        self.friction = float(friction)
        self.entity = None

    def create(self):
        import genesis as gs
        self.entity = self.scene.add_entity(
            gs.morphs.Box(
                size=(self.size,) * 3,
                pos=tuple(self.center),
            ),
            # High friction so the fingers can hold the cube while lifting.
            material=gs.materials.Rigid(friction=self.friction),
            surface=gs.surfaces.Default(color=self.color, opacity=1.0, roughness=0.4),
        )
        return self.entity

    def reset(self, center):
        self.center = np.array(center, dtype=np.float64)
        self.entity.set_pos(self.center)
        self.entity.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))
        self.entity.zero_all_dofs_velocity()

    def get_pos(self):
        pos = self.entity.get_pos()
        if hasattr(pos, 'detach'):
            pos = pos.detach().cpu().numpy()
        return np.asarray(pos, dtype=np.float64)
