'''Genesis initialization / scene construction for the CRANE-X7 environment.

Modeled after `samples/adrobo-CRANE-X7-main/simulation/config/genesis_init.py`
with fixes for the current Genesis API:
- `contact_resolve_time` is deprecated -> `constraint_timeconst`
- genesis is imported lazily so `import custom_envs` works without it
'''


class GenesisConfig:
    def __init__(
        self,
        device='cpu',
        seed=None,
        precision='64',
        logging_level='warning',
        show_viewer=False,
        dt=0.0025,
        show_cameras=False,
        plane_reflection=False,
        rigid_options=None,
    ):
        self.device = str(device)
        if self.device not in {'cpu', 'gpu'}:
            raise ValueError(f"Genesis device must be 'cpu' or 'gpu', got {self.device!r}")
        self.seed = seed
        self.precision = precision
        self.logging_level = logging_level
        self.show_viewer = bool(show_viewer)
        self.dt = float(dt)
        self.show_cameras = bool(show_cameras)
        self.plane_reflection = bool(plane_reflection)
        self.rigid_options = dict(rigid_options or {})
        self.scene = None

    def gs_init(self):
        import genesis as gs

        try:
            gs.init(
                seed=self.seed,
                precision=self.precision,
                logging_level=self.logging_level,
                backend=gs.cpu if self.device == 'cpu' else gs.gpu,
            )
        except Exception as exc:
            # gs.init can only run once per process; subsequent env creations reuse it.
            if 'already initialized' not in str(exc).lower():
                raise

        rigid = self.rigid_options
        contact_pruning_tolerance = rigid.get('contact_pruning_tolerance', None)
        if contact_pruning_tolerance is not None:
            contact_pruning_tolerance = float(contact_pruning_tolerance)
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                gravity=(0, 0, -9.81),
            ),
            rigid_options=gs.options.RigidOptions(
                enable_joint_limit=True,
                enable_collision=True,
                # The convexified finger collision meshes overlap each other,
                # blocking the gripper from closing; self-collision is not
                # needed for this task.
                enable_self_collision=bool(rigid.get('enable_self_collision', False)),
                constraint_solver=gs.constraint_solver.Newton,
                iterations=int(rigid.get('iterations', 400)),
                tolerance=float(rigid.get('tolerance', 1e-8)),
                ls_iterations=int(rigid.get('ls_iterations', 120)),
                noslip_iterations=int(rigid.get('noslip_iterations', 10)),
                constraint_timeconst=float(rigid.get('constraint_timeconst', 0.005)),
                contact_pruning_tolerance=contact_pruning_tolerance,
                max_collision_pairs=int(rigid.get('max_collision_pairs', 512)),
                use_gjk_collision=bool(rigid.get('use_gjk_collision', True)),
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=False,
                show_cameras=self.show_cameras,
                shadow=True,
                plane_reflection=self.plane_reflection,
                background_color=(0.02, 0.04, 0.08),
                ambient_light=(0.12, 0.12, 0.12),
                lights=[
                    {'type': 'directional', 'dir': (-0.6, -0.7, -1.0), 'color': (1.0, 0.98, 0.95), 'intensity': 3.0},
                    {'type': 'directional', 'dir': (0.4, 0.1, -1.0), 'color': (0.9, 0.95, 1.0), 'intensity': 1.5},
                ],
            ),
            show_viewer=self.show_viewer,
            renderer=gs.renderers.Rasterizer(),
        )
        return self.scene
