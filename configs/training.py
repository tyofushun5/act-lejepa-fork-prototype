import yaml
from box import Box
from pathlib import Path

def _create_include_loader(base_dir):
    """Creates a YAML loader that handles !inc directives."""
    class IncludeLoader(yaml.SafeLoader):
        pass

    def include(loader, node):
        """Include a YAML file."""
        filename = base_dir / loader.construct_scalar(node)
        with open(filename, 'r') as f:
            return yaml.load(f, IncludeLoader)

    IncludeLoader.add_constructor('!inc', include)
    return IncludeLoader

class Config(Box):
    def __repr__(self):
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def load(cls, file_path):
        base_dir = Path(file_path).resolve().parent
        Loader = _create_include_loader(base_dir)
        with open(file_path, 'r') as f:
            data = yaml.load(f, Loader=Loader)
        return cls(data)

    @classmethod
    def from_yaml_file(cls, file_path):
        return cls.load(file_path)

    @classmethod
    def from_json_file(cls, file_path):
        data = Box.from_json(filename=file_path).to_dict()
        return cls(data)