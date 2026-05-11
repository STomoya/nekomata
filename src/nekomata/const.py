"""Constants."""

# region Helpers

import os
from typing import overload


@overload
def get_env[EnvVarT](name: str, type: type[EnvVarT], default: EnvVarT) -> EnvVarT: ...


@overload
def get_env[EnvVarT](name: str, type: type[EnvVarT], default: None) -> EnvVarT | None: ...


def get_env[EnvVarT](name: str, type: type[EnvVarT], default: EnvVarT | None = None) -> EnvVarT | None:
    """Get environment variable and cast to desired type, or optionally fallback to defatul.

    Args:
        name (str): The environment variable name.
        type (type[EnvVarT]): The type to cast the envvar to.
        default (EnvVarT | None, optional): The default value.

    Returns:
        EnvVarT | None: The loaded evironment variable value.

    """
    value = os.getenv(key=name)
    if value:
        return type(value)
    return default


class Envs:
    """Package environment variable names."""

    _SUFFIX = 'NMATA_'
    UUID_VERSION = _SUFFIX + 'UUID_VERSION'


UUID_VERSION: int = get_env(Envs.UUID_VERSION, type=int, default=4)
