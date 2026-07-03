class Entity:
    '''Minimal base class for scene entities (mirrors the reference layout).'''

    def __init__(self, scene=None, surface=None):
        self.scene = scene
        self.surface = surface

    def create(self):
        raise NotImplementedError
