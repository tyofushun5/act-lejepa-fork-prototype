'''CRANE-X7 (Genesis) environment package.

Layout mirrors the reference implementation under
`samples/adrobo-CRANE-X7-main/simulation`:

- `config.py`  Genesis initialization / scene options
- `entity/`    robot, cube, camera, workspace entities
- `env.py`     the gymnasium environment + registration (CraneX7-v0)

genesis itself is imported lazily on env creation, so importing
`custom_envs` does not require it.
'''
from .entity.crane_x7 import GRIPPER_CLOSE, GRIPPER_OPEN, INIT_QPOS
from .env import CraneX7Env, make_crane_x7_env
