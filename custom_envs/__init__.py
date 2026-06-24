"""Register custom Gymnasium environments on package import."""

from . import metaworld as _metaworld
from . import pusht as _pusht
# ManiSkill can work on macOS, but GPU rendering is not supported out of the box.
from . import mani_skill as _mani_skill