import numpy as np

from .entity import Entity


class Workspace(Entity):
    '''Reachable end-effector workspace (bounds from the reference implementation).

    Unlike the reference, this is a plain bounds holder; the debug
    visualization box is created only when `create()` is called explicitly.
    '''

    def __init__(self, scene=None, surface=None, margin=0.0):
        super().__init__(scene=scene, surface=surface)
        # z_min lowered from the reference's 0.090: the fingertips reach about
        # 8 cm below the EE link, so grasping a ~2.5 cm cube on the table needs
        # EE heights around 0.085.
        self.workspace_min = np.array([0.150, -0.200, 0.075], dtype=np.float64)
        self.workspace_max = np.array([0.400, 0.200, 0.300], dtype=np.float64)
        self.workspace_margin = float(margin)

    @property
    def min_with_margin(self):
        return self.workspace_min + self.workspace_margin

    @property
    def max_with_margin(self):
        return self.workspace_max - self.workspace_margin

    def clip(self, pos):
        return np.clip(np.asarray(pos, dtype=np.float64), self.min_with_margin, self.max_with_margin)

    def create(self):
        import genesis as gs
        return self.scene.add_entity(
            gs.morphs.Box(
                lower=tuple(self.workspace_min),
                upper=tuple(self.workspace_max),
                visualization=True,
                collision=False,
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=(0.0, 1.0, 0.0), opacity=0.3),
        )
