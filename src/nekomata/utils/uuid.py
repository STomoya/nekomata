"""UUID utils."""

from fastuuid import uuid4, uuid7

from nekomata.const import UUID_VERSION


def fastuuid4() -> str:
    """UUID4."""
    return str(uuid4())


def fastuuid7() -> str:
    """UUID7."""
    return str(uuid7())


def create_uuid(version: int | str = UUID_VERSION) -> str:
    """Create UUID using fastuuid."""
    if int(version) == 4:  # noqa: PLR2004
        return fastuuid4()
    elif int(version) == 7:  # noqa: PLR2004
        return fastuuid7()
    else:
        raise ValueError(f'Unkown or unsupported UUID version: {version}')
