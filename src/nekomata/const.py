"""Constants."""

from nekomata.utils import get_env


class Envs:
    """Package environment variable names."""

    _SUFFIX = 'NMATA_'
    UUID_VERSION = _SUFFIX + 'UUID_VERSION'


UUID_VERSION: int = get_env(Envs.UUID_VERSION, type=int, default=4)
