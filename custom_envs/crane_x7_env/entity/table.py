import math
from pathlib import Path

from .entity import Entity


_TABLE_ASSET = Path(__file__).resolve().parents[1] / 'assets/table/table.glb'


class Table(Entity):
    '''Table asset from the CRANE-X7 sample environment.

    The rendered GLB is visual-only. The actual contact surface is a fixed box
    matching the sample implementation, with its tabletop at z=0.
    '''

    def __init__(self, scene=None, surface=None, offset=(0.5, 0.0, -0.9196429), scale=1.75):
        super().__init__(scene=scene, surface=surface)
        self.offset = tuple(offset)
        self.scale = float(scale)
        self._table_height = 0.9196429
        self.quat = (math.cos(-math.pi / 2), 0.0, 0.0, math.sin(-math.pi / 2))
        self.visual_entity = None
        self.entity = None

    @property
    def table_height(self):
        return self._table_height

    @property
    def table_path(self):
        return _TABLE_ASSET

    def create(self):
        import genesis as gs

        table_path = self.table_path
        if not table_path.exists():
            raise FileNotFoundError(f'CRANE-X7 table asset not found: {table_path}')

        visual = gs.morphs.Mesh(
            file=str(table_path),
            scale=self.scale,
            pos=self.offset,
            quat=self.quat,
            fixed=True,
            collision=False,
            # Equivalent to the sample's deprecated parse_glb_with_zup=False.
            file_meshes_are_zup=True,
        )
        self.visual_entity = self.scene.add_entity(
            morph=visual,
            material=None,
            surface=self.surface,
            visualize_contact=False,
            vis_mode='visual',
        )

        half_x = 1.209 / 2
        half_y = 2.418 / 2
        lower = (
            self.offset[0] - half_x,
            self.offset[1] - half_y,
            self.offset[2],
        )
        upper = (
            self.offset[0] + half_x,
            self.offset[1] + half_y,
            self.offset[2] + self._table_height,
        )
        collision = gs.morphs.Box(
            lower=lower,
            upper=upper,
            visualization=False,
            collision=True,
            fixed=True,
        )
        self.entity = self.scene.add_entity(
            morph=collision,
            material=None,
            surface=self.surface,
            visualize_contact=False,
            vis_mode='visual',
        )
        return self.entity
